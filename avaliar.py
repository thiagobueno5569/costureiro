"""Avaliação em tarefas de verdade: a IA costurada contra cada IA original, nas mesmas condições.

    python avaliar.py --ia ias_novas/codigo-mais-matematica
    python avaliar.py --ia ias_novas/minha-ia --tarefas humaneval,gsm8k --n 50

Tarefas (as mesmas que a área usa):
  humaneval  164 funções Python; a resposta passa nos testes oficiais?        (pass@1)
  mbpp       257 problemas Python curtos (versão "sanitized"); passa nos asserts? (pass@1)
  gsm8k      problemas de matemática do ensino fundamental; o número final bate?  (acerto)

Regras iguais para todos: decodificação gulosa (sempre a palavra mais provável), sem penalidade de
repetição, mesmos prompts, mesmos limites de tamanho e mesmas paradas.

⚠️ HumanEval e MBPP EXECUTAM o código gerado pelas IAs (cada um num processo separado, com tempo
limite, numa pasta temporária). É o procedimento padrão, mas prefira rodar num servidor (RunPod etc.).
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

import torch

import nucleo

AQUI = Path(__file__).parent
DADOS = AQUI / "dados_avaliacao"
URLS = {
    "humaneval": "https://huggingface.co/datasets/openai/openai_humaneval/resolve/main/openai_humaneval/test-00000-of-00001.parquet",
    "mbpp": "https://huggingface.co/datasets/google-research-datasets/mbpp/resolve/main/sanitized/test-00000-of-00001.parquet",
    "gsm8k": "https://huggingface.co/datasets/openai/gsm8k/resolve/main/main/test-00000-of-00001.parquet",
}
PARADAS = {
    "humaneval": ["\ndef ", "\nclass ", "\nif __name__", "\nprint(", "\n#", "\n```", "\nassert"],
    "mbpp": ['\n"""', "\nassert", "\nif __name__", "\nprint(", "\n```", "\n# Test", "\n\n\n\n"],
    "gsm8k": ["\n\nProblem:", "\nProblem:", "\n\n\n"],
}
TAMANHO = {"humaneval": 384, "mbpp": 384, "gsm8k": 320}


# ============================================================================ dados

def carregar_tarefa(nome: str) -> list[dict]:
    import pandas as pd
    DADOS.mkdir(exist_ok=True)
    arq = DADOS / f"{nome}.parquet"
    if not arq.exists():
        req = urllib.request.Request(URLS[nome], headers={"User-Agent": "costureiro"})
        arq.write_bytes(urllib.request.urlopen(req, timeout=180).read())
    df = pd.read_parquet(arq)
    itens = []
    for r in df.to_dict("records"):
        if nome == "humaneval":
            itens.append({"id": r["task_id"], "prompt": r["prompt"], "teste": r["test"], "entrada": r["entry_point"]})
        elif nome == "mbpp":
            testes = list(r["test_list"])
            itens.append({"id": str(r["task_id"]), "prompt": f'"""\n{r["prompt"]}\n{testes[0]}\n"""\n',
                          "teste": "\n".join(list(r["test_imports"]) + testes)})
        else:
            gab = r["answer"].split("####")[-1].strip().replace(",", "")
            itens.append({"id": str(len(itens)), "prompt": f"Problem: {r['question']}\nSolution:", "gabarito": gab})
    return itens


# ============================================================================ geração

def cortar(texto: str, paradas: list[str]) -> str:
    cortes = [texto.find(p) for p in paradas if texto.find(p) >= 0]
    return texto[:min(cortes)] if cortes else texto


class Original:
    def __init__(self, nome):
        self.nome = nome
        self.tok, self.m = nucleo.carregar_modelo(nome)

    @torch.no_grad()
    def completar(self, prompt, max_novos, paradas):
        x = self.tok(prompt, return_tensors="pt", add_special_tokens=False).to(next(self.m.parameters()).device)
        from transformers import StoppingCriteria, StoppingCriteriaList
        n0, tok = x["input_ids"].shape[1], self.tok

        class Para(StoppingCriteria):                  # para quando o texto novo contém uma parada
            def __call__(self, ids, scores, **kw):
                return any(p in tok.decode(ids[0][n0:], skip_special_tokens=True) for p in paradas)
        o = self.m.generate(**x, max_new_tokens=max_novos, do_sample=False, pad_token_id=self.tok.eos_token_id,
                            stopping_criteria=StoppingCriteriaList([Para()]))
        return cortar(self.tok.decode(o[0][x["input_ids"].shape[1]:], skip_special_tokens=True), paradas)


class Costurada:
    def __init__(self, pasta):
        self.ia = nucleo.IACosturada.abrir(Path(pasta), log=lambda *_: None)
        self.ia.r.max_tok = 4096          # sem cortar o contexto (o tradutor foi treinado com trechos curtos)
        self.nome = f"costurada ({Path(pasta).name})"

    @torch.no_grad()
    def completar(self, prompt, max_novos, paradas):
        tok = self.ia.mods[-1][0]
        gerados, texto = [], ""
        for _ in range(max_novos):
            d = self.ia.preparar(prompt + texto)
            if d is None:
                break
            prox = int(self.ia._cadeia(d)[-1].argmax())
            if prox == tok.eos_token_id:
                break
            gerados.append(prox)
            texto = tok.decode(gerados, skip_special_tokens=True).rstrip("�")
            if any(p in texto for p in paradas):
                break
        return cortar(texto, paradas)


class Cerebro:
    """IA do modo cérebro: a base consulta os especialistas pelo portão."""

    def __init__(self, pasta):
        self.ia = nucleo.IACerebro.abrir(Path(pasta), log=lambda *_: None)
        self.nome = f"cérebro ({Path(pasta).name})"

    @torch.no_grad()
    def completar(self, prompt, max_novos, paradas):
        tok = self.ia.tok
        ids = tok(prompt, add_special_tokens=False)["input_ids"]
        gerados, texto = [], ""
        for _ in range(max_novos):
            prox = int(self.ia.logits(ids[-2048:])[-1].argmax())
            if prox == tok.eos_token_id:
                break
            ids.append(prox); gerados.append(prox)
            texto = tok.decode(gerados, skip_special_tokens=True).rstrip("\ufffd")
            if any(p in texto for p in paradas):
                break
        return cortar(texto, paradas)


class Orquestra:
    def __init__(self, pasta):
        import orquestra
        self.ia = orquestra.IAOrquestra.abrir(Path(pasta), log=lambda *_: None)
        self.nome = f"orquestra ({Path(pasta).name})"

    def completar(self, prompt, max_novos, paradas):
        return self.ia.completar(prompt, max_novos, paradas)


# ============================================================================ correção

def rodar_codigo(programa: str, tempo: float = 10.0) -> bool:
    with tempfile.TemporaryDirectory() as d:
        arq = Path(d) / "teste.py"
        arq.write_text(programa, encoding="utf-8")
        try:
            p = subprocess.run([sys.executable, "-I", str(arq)], cwd=d, capture_output=True, timeout=tempo)
            return p.returncode == 0
        except subprocess.TimeoutExpired:
            return False


def numero_final(texto: str) -> str | None:
    m = re.search(r"Answer:\s*\$?(-?[\d,]*\.?\d+)", texto)
    nums = [m.group(1)] if m else re.findall(r"-?[\d,]*\.?\d+", texto)
    if not nums:
        return None
    return nums[-1].replace(",", "").rstrip(".")


def corrigir(tarefa: str, item: dict, resposta: str) -> bool:
    if tarefa == "humaneval":
        return rodar_codigo(item["prompt"] + resposta + "\n\n" + item["teste"] + f"\n\ncheck({item['entrada']})\n")
    if tarefa == "mbpp":
        return rodar_codigo(resposta + "\n\n" + item["teste"] + "\n")
    n = numero_final(resposta)
    try:
        return n is not None and abs(float(n) - float(item["gabarito"])) < 1e-6
    except ValueError:
        return False


# ============================================================================ tudo

def avaliar(pasta_ia: Path, tarefas: list[str], n: int = 0, extras: list[str] = (), log=print,
            saida: Path | None = None, so_originais: bool = False) -> dict:
    pasta_ia = Path(pasta_ia)
    if not (pasta_ia / "receita.json").exists():
        raise SystemExit(f"não achei {pasta_ia / 'receita.json'} — use a pasta de uma IA em ias_novas/")
    cfg = json.loads((pasta_ia / "receita.json").read_text(encoding="utf-8"))
    if cfg.get("tipo") == "orquestra":                # orquestra: compara com a base e com cada solista
        import orquestra
        r = orquestra.ReceitaOrquestra.abrir(pasta_ia)
        modelos = [] if so_originais else [("orquestra", lambda: Orquestra(pasta_ia))]
        originais = [r.base] + [s.modelo for s in r.solistas]
        partes = [{"modelo": r.base, "inicio": 0, "fim": "base"}] + \
                 [{"modelo": s.modelo, "inicio": 0, "fim": s.ate} for s in r.solistas]
    elif cfg.get("tipo") == "cerebro":                  # modo cérebro: compara com a base e com cada especialista
        r = nucleo.ReceitaCerebro.abrir(pasta_ia)
        modelos = [] if so_originais else [("cerebro", lambda: Cerebro(pasta_ia))]
        originais = [r.base] + [e.modelo for e in r.especialistas]
        partes = [{"modelo": r.base, "inicio": 0, "fim": "base"}] + \
                 [{"modelo": e.modelo, "inicio": e.inicio, "fim": e.fim} for e in r.especialistas]
    else:
        r = nucleo.Receita.abrir(pasta_ia)
        modelos = [] if so_originais else [("costurada", lambda: Costurada(pasta_ia))]
        originais = [p.modelo for p in r.partes]
        partes = [p.__dict__ for p in r.partes]
    for nm in dict.fromkeys(originais):
        modelos.append((nm, lambda nm=nm: Original(nm)))
    for e in extras:
        modelos.append((e, lambda nm=e: Original(nm)))
    dados = {t: carregar_tarefa(t)[: n or None] for t in tarefas}
    res = {"ia": str(pasta_ia), "partes": partes, "tarefas": {}, "detalhes": {}}
    for chave, fabrica in modelos:
        nucleo.liberar_cache()
        m = fabrica()
        for t in tarefas:
            t0, certos, det = time.time(), 0, []
            for k, item in enumerate(dados[t]):
                resp = m.completar(item["prompt"], TAMANHO[t], PARADAS[t])
                ok = corrigir(t, item, resp)
                certos += ok
                det.append({"id": item["id"], "ok": ok, "resposta": resp[:2000]})
                if (k + 1) % 10 == 0 or k + 1 == len(dados[t]):
                    log(f"  {m.nome:40s} {t:9s} {k + 1:4d}/{len(dados[t])}  {100 * certos / (k + 1):5.1f}%  "
                        f"({(time.time() - t0) / (k + 1):.1f} s por item)")
            res["tarefas"].setdefault(t, {})[m.nome] = {"acerto": certos / max(1, len(dados[t])), "n": len(dados[t]),
                                                         "s_por_item": (time.time() - t0) / max(1, len(dados[t]))}
            res["detalhes"].setdefault(t, {})[m.nome] = det
            if saida:
                salvar(res, saida)
        del m
    nucleo.liberar_cache()
    referencias(res, originais, dados)
    if saida:
        salvar(res, saida)
    return res


def mcnemar(a: list[bool], b: list[bool]) -> tuple[int, int, float]:
    """Teste pareado exato: (só a acertou, só b acertou, p bilateral). p < 0,05 = diferença provavelmente real."""
    from math import comb
    x = sum(1 for u, v in zip(a, b) if u and not v)
    y = sum(1 for u, v in zip(a, b) if v and not u)
    n, k = x + y, min(x, y)
    p = 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)
    return x, y, p


def parece_codigo(prompt: str) -> bool:
    return "def " in prompt or '"""' in prompt or "import " in prompt


def referencias(res: dict, originais: list[str], dados: dict):
    """Duas réguas que qualquer fusão precisa vencer para valer a pena:
    roteador simples: escolhe a IA ORIGINAL inteira pela cara da pergunta (código → a de código; resto → a outra);
    oráculo: conta acerto se QUALQUER original acertou (teto de 'escolher a IA certa por pergunta')."""
    orig = list(dict.fromkeys(originais))
    cod = next((m for m in orig if "coder" in m.lower() or "code" in m.lower()), None)
    outra = next((m for m in orig if m != cod), orig[0])
    for t, por_modelo in res["detalhes"].items():
        if not all(m in por_modelo for m in orig):
            continue
        itens = dados[t]
        acert_r, acert_o = 0, 0
        for k, item in enumerate(itens):
            escolhido = cod if (cod and parece_codigo(item["prompt"])) else outra
            acert_r += por_modelo[escolhido][k]["ok"]
            acert_o += any(por_modelo[m][k]["ok"] for m in orig)
        n = max(1, len(itens))
        res["tarefas"][t]["roteador simples (IA inteira por pergunta)"] = {"acerto": acert_r / n, "n": len(itens), "s_por_item": 0}
        rot = [por_modelo[cod if (cod and parece_codigo(it["prompt"])) else outra][k]["ok"] for k, it in enumerate(itens)]
        for nova in [m for m in por_modelo if m.startswith(("costurada", "cérebro", "orquestra"))]:
            ok_n = [x["ok"] for x in por_modelo[nova]]
            comp = {m: mcnemar(ok_n, [x["ok"] for x in por_modelo[m]]) for m in orig}
            comp["roteador"] = mcnemar(ok_n, rot)
            res.setdefault("pareado", {}).setdefault(t, {})[nova] = comp
        res["tarefas"][t]["oráculo (alguma original acerta)"] = {"acerto": acert_o / n, "n": len(itens), "s_por_item": 0}


def relatorio(res: dict) -> str:
    nomes = {"humaneval": "HumanEval (pass@1)", "mbpp": "MBPP sanitized (pass@1)", "gsm8k": "GSM8K (acerto)"}
    modelos = list(dict.fromkeys(m for t in res["tarefas"].values() for m in t))
    lin = [f"# Avaliação em tarefas de verdade · `{Path(res['ia']).name}`", "",
           "Partes: " + " · ".join(f"`{p['modelo']}` " + ("(base)" if p["fim"] == "base" else f"[{p['inicio']}–{p['fim']}]")
                                   for p in res["partes"]), "",
           "| modelo | " + " | ".join(nomes[t] for t in res["tarefas"]) + " |",
           "|---|" + "---|" * len(res["tarefas"])]
    for m in modelos:
        cel = []
        for t, v in res["tarefas"].items():
            x = v.get(m)
            cel.append(f"{100 * x['acerto']:.1f}% (n={x['n']})" if x else "—")
        nova = m.startswith(("costurada", "cérebro", "orquestra"))
        regua = m.startswith(("roteador", "oráculo"))
        rotulo = f"**{m}**" if nova else (f"*{m}*" if regua else f"`{m}`")
        lin.append(f"| {rotulo} | " + " | ".join(cel) + " |")
    if res.get("pareado"):
        lin += ["", "**Teste pareado (McNemar exato)**: em quantos itens só a IA nova acertou × só a outra acertou, "
                "e a chance de a diferença ser sorte (p < 0,05 = provavelmente real):", ""]
        for t, por_nova in res["pareado"].items():
            for nova, comps in por_nova.items():
                lin.append(f"- {nomes[t]} · {nova}: " + "; ".join(
                    f"contra {Path(m).name}: +{x}/−{y}, p = {p:.3f}" for m, (x, y, p) in comps.items()))
    lin += ["", "Decodificação gulosa, sem penalidade de repetição, mesmos prompts e paradas para todos. "
            "HumanEval/MBPP: o código gerado roda contra os testes oficiais. GSM8K: compara o número final.",
            "*Roteador simples*: escolhe a IA original inteira pela cara da pergunta (código → a de código). "
            "*Oráculo*: acerto se alguma original acertou (o teto de escolher a IA certa). "
            "Uma fusão só vale a pena se passar do roteador.",
            "", "Respostas de cada item: arquivo `.json` ao lado."]
    return "\n".join(lin)


def salvar(res: dict, saida: Path):
    saida = Path(saida)
    saida.parent.mkdir(parents=True, exist_ok=True)
    saida.with_suffix(".json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    saida.with_suffix(".md").write_text(relatorio(res), encoding="utf-8")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Costureiro · avaliação em tarefas de verdade")
    ap.add_argument("--ia", required=True, help="pasta da IA nova (ias_novas/...): costurada ou do modo cérebro")
    ap.add_argument("--tarefas", default="humaneval,mbpp,gsm8k")
    ap.add_argument("--n", type=int, default=0, help="quantos itens por tarefa (0 = todos)")
    ap.add_argument("--extra", action="append", default=[], help="outro modelo para comparar (repita)")
    ap.add_argument("--saida", default=None)
    a = ap.parse_args(argv)
    pasta = Path(a.ia)
    saida = Path(a.saida) if a.saida else AQUI / "resultados" / f"avaliacao_{pasta.name}"
    res = avaliar(pasta, [t.strip() for t in a.tarefas.split(",") if t.strip()], a.n, a.extra,
                  log=lambda m: print(m, flush=True), saida=saida)
    print("\n" + relatorio(res))
    print(f"\nsalvo em {saida.with_suffix('.md')}")


if __name__ == "__main__":
    main()
