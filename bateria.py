"""Bateria de testes no servidor, usando a placa inteira: treina e avalia várias IAs ao mesmo tempo.

Um "gerente" lança cada trabalho quando há memória livre na placa (em qualquer placa, se houver várias),
respeita a ordem (avaliar só depois de treinar), divide as tarefas longas em pedaços paralelos e, no fim,
junta tudo num relatório só com o teste pareado. Opcional: desliga o pod (RunPod) quando termina.

    python bateria.py --desligar
Acompanhe:  tail -f resultados/bateria/gerente.log
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

AQUI = Path(__file__).parent
SAIDA = AQUI / "resultados" / "bateria"
PY = sys.executable
CODER = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
MATH = "Qwen/Qwen2.5-Math-1.5B-Instruct"


def log(msg):
    linha = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(linha, flush=True)
    with open(SAIDA / "gerente.log", "a", encoding="utf-8") as f:
        f.write(linha + "\n")


def placas_livres() -> list[tuple[int, float]]:
    """(índice, GB livres) de cada placa."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=20).stdout
        return [(int(a), float(b) / 1024) for a, b in (l.split(",") for l in out.strip().splitlines())]
    except Exception:
        return [(0, 0.0)]


def trabalhos(n_gsm: int, pedacos_gsm: int, epocas: int, trechos: int) -> list[dict]:
    T = []

    def treino(nome, base, solista, ep, tr):
        T.append({"nome": f"treino_{nome}", "gb": 13, "dep": [],
                  "cmd": [PY, "orquestra.py", "--base", base, "--solista", solista, "--nome", nome,
                          "--epocas", str(ep), "--max_trechos", str(tr)],
                  "pular_se": AQUI / "ias_novas" / nome / "ponte.pt"})

    def avaliacoes(nome, com_originais, dep):
        extra = [] if com_originais else ["--sem-originais"]
        for t in ("humaneval", "mbpp"):
            T.append({"nome": f"aval_{nome}_{t}", "gb": 7, "dep": dep,
                      "cmd": [PY, "avaliar.py", "--ia", f"ias_novas/{nome}", "--tarefas", t, *extra,
                              "--saida", str(SAIDA / f"{nome}_{t}")]})
        for k in range(pedacos_gsm):
            T.append({"nome": f"aval_{nome}_gsm8k_{k}", "gb": 7, "dep": dep,
                      "cmd": [PY, "avaliar.py", "--ia", f"ias_novas/{nome}", "--tarefas", "gsm8k", "--n", str(n_gsm),
                              "--fatia", f"{k}/{pedacos_gsm}", *extra, "--saida", str(SAIDA / f"{nome}_gsm8k_{k}")]})

    # treinos primeiro na fila (precisam de mais memória; as avaliações preenchem o resto da placa)
    # controle: mesmo treino, mas o solista é o PRÓPRIO Coder. Separa "o Math ajudou" de "o treino no formato ajudou"
    treino("orq-coder-com-coder", CODER, CODER, epocas, trechos)
    # mais treino e mais dados (a matemática melhora com mais exemplos?)
    treino("orq-coder-com-math-longo", CODER, MATH, epocas + 1, trechos * 2)
    # a orquestra que já existe (treinada ontem): avaliação completa, com os originais
    avaliacoes("orq-coder-com-math", True, [])
    avaliacoes("orq-coder-com-coder", False, ["treino_orq-coder-com-coder"])
    avaliacoes("orq-coder-com-math-longo", False, ["treino_orq-coder-com-math-longo"])
    return T


def preparar():
    """Baixa dados e modelos UMA vez antes de tudo (vários processos baixando juntos se atrapalham)."""
    log("preparando: dados das tarefas e modelos")
    subprocess.run([PY, "-c", "import avaliar\nfor t in ('humaneval','mbpp','gsm8k'): avaliar.carregar_tarefa(t)"],
                   cwd=AQUI, check=True)
    subprocess.run([PY, "-c", f"from huggingface_hub import snapshot_download\nfor m in ('{CODER}','{MATH}'): "
                              f"snapshot_download(m, allow_patterns=['*.json','*.safetensors','*.txt','*.model','*.jinja'])"],
                   cwd=AQUI, check=True)
    destino = AQUI / "ias_novas" / "orq-coder-com-math"
    if not (destino / "ponte.pt").exists() and (AQUI / "modelos" / "orq-coder-com-math").exists():
        shutil.copytree(AQUI / "modelos" / "orq-coder-com-math", destino, dirs_exist_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-gsm8k", type=int, default=300)
    ap.add_argument("--pedacos-gsm8k", type=int, default=3)
    ap.add_argument("--epocas", type=int, default=2)
    ap.add_argument("--trechos", type=int, default=1500)
    ap.add_argument("--max-juntos", type=int, default=8)
    ap.add_argument("--desligar", action="store_true", help="desliga o pod do RunPod no fim (Stop)")
    a = ap.parse_args()
    SAIDA.mkdir(parents=True, exist_ok=True)
    os.chdir(AQUI)
    preparar()
    fila = trabalhos(a.n_gsm8k, a.pedacos_gsm8k, a.epocas, a.trechos)
    feitos, falhou, rodando = set(), set(), {}
    for j in fila:
        if j.get("pular_se") and Path(j["pular_se"]).exists():
            feitos.add(j["nome"]); log(f"pulando {j['nome']} (já existe)")
    t0 = time.time()
    while True:
        # quem terminou?
        for nome, (p, arq, inicio) in list(rodando.items()):
            if p.poll() is not None:
                arq.close()
                del rodando[nome]
                if p.returncode == 0:
                    feitos.add(nome); log(f"✓ {nome} ({(time.time() - inicio) / 60:.0f} min)")
                else:
                    falhou.add(nome); log(f"✗ {nome} falhou (código {p.returncode}); veja {SAIDA / (nome + '.log')}")
        pendentes = [j for j in fila if j["nome"] not in feitos | falhou | set(rodando)]
        for j in pendentes:
            if any(d in falhou for d in j["dep"]):
                falhou.add(j["nome"]); log(f"✗ {j['nome']} pulado (dependia de algo que falhou)")
        pendentes = [j for j in pendentes if j["nome"] not in falhou]
        if not pendentes and not rodando:
            break
        lancou = False
        for j in pendentes:
            if not all(d in feitos for d in j["dep"]) or len(rodando) >= a.max_juntos:
                continue
            placa, livre = max(placas_livres(), key=lambda x: x[1])
            if livre < j["gb"] + 1:
                continue
            arq = open(SAIDA / f"{j['nome']}.log", "w", encoding="utf-8")
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(placa), PYTHONUNBUFFERED="1",
                       PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
            p = subprocess.Popen(j["cmd"], stdout=arq, stderr=subprocess.STDOUT, env=env, cwd=AQUI)
            rodando[j["nome"]] = (p, arq, time.time())
            log(f"→ {j['nome']} na placa {placa} ({livre:.0f} GB livres; {len(rodando)} rodando)")
            lancou = True
            break                      # um por vez: espera a memória dele aparecer antes do próximo
        if lancou:
            time.sleep(75)
        else:
            time.sleep(20)
            if int(time.time() - t0) % 600 < 20:
                log(f"rodando: {', '.join(rodando) or '-'} · faltam {len(pendentes)} · feitos {len(feitos)} · "
                    f"placas: {', '.join(f'{i}: {g:.0f} GB livres' for i, g in placas_livres())}")

    log(f"todos os trabalhos acabaram em {(time.time() - t0) / 60:.0f} min ({len(feitos)} ok, {len(falhou)} com falha)")
    jsons = sorted(str(p) for p in SAIDA.glob("*.json"))
    if jsons:
        subprocess.run([PY, "avaliar.py", "--juntar", *jsons, "--saida", str(SAIDA / "RESULTADO")], cwd=AQUI)
        log(f"relatório: {SAIDA / 'RESULTADO.md'}")
        print((SAIDA / "RESULTADO.md").read_text(encoding="utf-8"), flush=True)
    if a.desligar and os.environ.get("RUNPOD_POD_ID") and shutil.which("runpodctl"):
        log("desligando o pod (Stop: o /workspace fica guardado)")
        time.sleep(20)
        subprocess.run(["runpodctl", "stop", "pod", os.environ["RUNPOD_POD_ID"]])


if __name__ == "__main__":
    main()
