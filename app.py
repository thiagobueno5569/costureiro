"""
Costureiro — crie uma IA nova juntando as melhores partes de outras IAs.

    python app.py        (abre no navegador em http://127.0.0.1:7861)
"""
import json
import queue
import random
import sys
import threading
import traceback
from pathlib import Path

import gradio as gr
import plotly.graph_objects as go

sys.path.insert(0, str(Path(__file__).parent))
import nucleo
from nucleo import Especialista, IACerebro, IACosturada, Parte, Receita, ReceitaCerebro

RAIZ = Path(__file__).resolve().parent
SAIDAS = RAIZ / "ias_novas"
TEXTOS = RAIZ / "textos"
FUSOES = RAIZ / "fusoes"
SAIDAS.mkdir(exist_ok=True)

SUGESTOES = [
    "Qwen/Qwen2.5-Coder-0.5B-Instruct", "Qwen/Qwen2.5-Coder-1.5B-Instruct", "Qwen/Qwen2.5-Math-1.5B-Instruct",
    "Qwen/Qwen2.5-0.5B-Instruct", "Qwen/Qwen2.5-1.5B-Instruct", "Qwen/Qwen3-0.6B", "Qwen/Qwen3-1.7B",
    "HuggingFaceTB/SmolLM2-360M-Instruct", "HuggingFaceTB/SmolLM2-1.7B-Instruct",
    "Qwen/Qwen2.5-3B-Instruct", "Qwen/Qwen2.5-Coder-3B-Instruct", "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen2.5-Coder-7B-Instruct", "Qwen/Qwen2.5-Math-7B-Instruct",
]
# paleta categórica (validada para fundo escuro: luminosidade, croma, daltonismo e contraste)
COR = ["#3987e5", "#d95926", "#199e70"]
SUPERFICIE, TEXTO, TEXTO2, GRADE = "#1a1a19", "#f0efec", "#c3c2b7", "#2c2c2a"
MAX_PARTES, MAX_ASSUNTOS = 3, 3
_parar = threading.Event()

CSS = """
body, .gradio-container { background: #0e0e0d !important; }
.gradio-container { max-width: 1180px !important; margin: auto; }
#topo { padding: 28px 8px 8px; }
#topo h1 { font-size: 30px; letter-spacing: -0.5px; margin: 0; color: #f0efec; }
#topo p { color: #c3c2b7; margin: 6px 0 0; font-size: 15px; }
.passo { color: #8b8a83; font-size: 12px; letter-spacing: 1.5px; text-transform: uppercase; margin: 4px 0 6px; }
.cartao { background: #1a1a19; border: 1px solid #2c2c2a; border-radius: 14px; padding: 12px 16px; color: #f0efec; }
.selo { display: inline-block; padding: 2px 9px; border-radius: 999px; font-size: 12px; border: 1px solid #3a3a37; color: #c3c2b7; margin-left: 6px; }
.costura { display: flex; gap: 6px; align-items: stretch; margin: 6px 0 4px; }
.costura .bloco { border-radius: 10px; padding: 10px 12px; color: #0e0e0d; font-weight: 600; min-width: 70px; }
.costura .ponte { align-self: center; color: #f0efec; font-size: 13px; white-space: nowrap; }
.costura small { display: block; font-weight: 500; opacity: .85; }
.recomenda { border-left: 3px solid #3987e5; padding: 4px 14px; }
footer { display: none !important; }
.prose code, .md code { background: #232321 !important; color: #f0efec !important; border: none !important; }
.prose table, .prose th, .prose td { border-color: #2c2c2a !important; }
"""


# ----------------------------------------------------------------------------- utilidades

def info(nome):
    try:
        return nucleo.info_modelo(nome.strip())
    except Exception:
        return None


def nome_curto(n):
    return n.split("/")[-1] if n else ""


def ler_assuntos(*v):
    out = {}
    for j in range(MAX_ASSUNTOS):
        a, c = v[2 * j], v[2 * j + 1]
        if a and c:
            out[a.strip()] = [x.strip().strip('"') for x in c.split(";") if x.strip()]
    return out


def diagrama(n, *v):
    """Desenho da IA nova: blocos proporcionais às camadas usadas de cada IA."""
    n = int(n)
    modelos, lim = v[:MAX_PARTES], v[MAX_PARTES:MAX_PARTES * 3]
    total = sum(max(1, int(lim[2 * i + 1] or 0) - int(lim[2 * i] or 0) + 1) for i in range(n))
    pedacos = []
    for i in range(n):
        if not modelos[i]:
            continue
        ini, fim = int(lim[2 * i] or 0), int(lim[2 * i + 1] or 0)
        larg = max(8, 100 * max(1, fim - ini + 1) / max(total, 1))
        papel = "lê" if i == 0 else ("escreve" if i == n - 1 else "pensa")
        pedacos.append(f'<div class="bloco" style="flex:{larg:.0f};background:{COR[i]}">{nome_curto(modelos[i])}'
                       f'<small>camadas {ini}–{fim} · {papel}</small></div>')
        if i < n - 1:
            pedacos.append('<div class="ponte">⇄ tradutor</div>')
    return ('<div class="passo">a IA nova</div><div class="costura">' + "".join(pedacos) + "</div>"
            '<div style="color:#8b8a83;font-size:12px">o texto entra à esquerda e a resposta sai à direita. '
            'Cada ⇄ é um super agente que aprende a mudança de base entre as duas.</div>')


def cartoes_modelos(n, *modelos):
    n = int(n)
    html = []
    for i in range(n):
        m = modelos[i]
        if not m:
            continue
        inf = info(m)
        papel = "lê o texto" if i == 0 else ("escreve a resposta" if i == n - 1 else "pensa no meio")
        det = (f'{inf["camadas"]} camadas · vetor de {inf["dimensao"]} · {inf["arquitetura"]}' if inf
               else "não consegui ler a configuração (nome certo? internet?)")
        html.append(f'<div class="cartao" style="border-left:4px solid {COR[i]};margin-bottom:8px">'
                    f'<b>IA {i + 1} · {nome_curto(m)}</b><span class="selo">{papel}</span>'
                    f'<div style="color:#c3c2b7;font-size:13px;margin-top:4px">{det}</div></div>')
    return "".join(html)


def ajustar_limites(n, *modelos):
    """Sliders com o máximo certo de camadas e uma divisão inicial proporcional."""
    n = int(n)
    out = []
    for i in range(MAX_PARTES):
        inf = info(modelos[i]) if i < n and modelos[i] else None
        if not inf:
            out += [gr.update(visible=False), gr.update(visible=False)]
            continue
        L = inf["camadas"]
        ini = 0 if i == 0 else round(i / n * L)
        fim = L - 1 if i == n - 1 else round((i + 1) / n * L) - 1
        out += [gr.update(visible=True, maximum=L - 1, value=ini, interactive=i != 0,
                          label=f"IA {i + 1} · {nome_curto(modelos[i])}: da camada"),
                gr.update(visible=True, maximum=L - 1, value=fim, interactive=i != n - 1, label="até a camada")]
    return out


# ----------------------------------------------------------------------------- gráficos

def _layout(fig, titulo, altura=330):
    fig.update_layout(title=dict(text=titulo, font=dict(size=15, color=TEXTO), x=0, xanchor="left"),
                      paper_bgcolor=SUPERFICIE, plot_bgcolor=SUPERFICIE, height=altura,
                      font=dict(color=TEXTO2, size=12), margin=dict(l=60, r=20, t=56, b=44),
                      legend=dict(orientation="h", y=1.14, x=1, xanchor="right", font=dict(color=TEXTO2)),
                      hoverlabel=dict(bgcolor="#232321", font_color=TEXTO))
    fig.update_xaxes(gridcolor=GRADE, zerolinecolor=GRADE, linecolor=GRADE)
    fig.update_yaxes(gridcolor=GRADE, zerolinecolor=GRADE, linecolor=GRADE)
    return fig


def grafico_habilidade(res):
    fig = go.Figure()
    for i, (nome, d) in enumerate(res["modelos"].items()):
        fig.add_bar(name=nome_curto(nome), x=res["assuntos"], y=[d["bpc"][a] for a in res["assuntos"]],
                    marker=dict(color=COR[i % 3], cornerradius=4, line=dict(width=0)),
                    text=[f"{d['bpc'][a]:.2f}" for a in res["assuntos"]], textposition="outside",
                    textfont=dict(color=TEXTO2),
                    hovertemplate="%{x}<br>" + nome_curto(nome) + ": %{y:.3f} bits/caractere<extra></extra>")
    topo = max(d["bpc"][a] for d in res["modelos"].values() for a in res["assuntos"])
    fig.update_layout(barmode="group", bargap=0.35, bargroupgap=0.08)
    fig.update_yaxes(title="bits por caractere (menor = melhor)", range=[0, topo * 1.18])
    return _layout(fig, "No que cada IA é boa")


def grafico_camadas(res, nome, i):
    d = res["modelos"][nome]
    z = [[max(0.0, v) for v in d["importancia"][a]] for a in res["assuntos"]]
    # a camada 0 costuma ser vital para tudo e apagaria o resto da escala: a cor satura no 90º percentil
    todos = sorted(v for linha in z for v in linha)
    zmax = todos[int(0.9 * (len(todos) - 1))] if todos else 1
    fig = go.Figure(go.Heatmap(
        z=z, zmin=0, zmax=max(zmax, 1e-6), x=list(range(d["camadas"])), y=res["assuntos"], colorscale=[[0, SUPERFICIE], [1, COR[i % 3]]],
        xgap=2, ygap=2, colorbar=dict(title=dict(text="piora sem ela", side="right"), thickness=10),
        customdata=[[max(0.0, v) for v in d["importancia"][a]] for a in res["assuntos"]],
        hovertemplate="camada %{x}<br>%{y}: sem ela, +%{customdata:.3f} bits/caractere<extra></extra>"))
    fig.update_xaxes(title="camada (0 = começo)", dtick=2)
    return _layout(fig, f"Melhores partes de {nome_curto(nome)}: quanto mais forte a cor, mais a camada importa para o assunto", 240)


def grafico_cortes(curva):
    xs = [int(100 * c["fracao"]) for c in curva]
    ys = [c["bpc"] for c in curva]
    k = ys.index(min(ys))
    fig = go.Figure()
    fig.add_scatter(x=xs, y=ys, mode="lines+markers", line=dict(color=COR[0], width=2),
                    marker=dict(size=9, color=COR[0], line=dict(color=SUPERFICIE, width=2)),
                    customdata=[[c["fim_a"], c["inicio_b"]] for c in curva],
                    hovertemplate="corte em %{x}%<br>IA 1 até a camada %{customdata[0]}, IA 2 a partir da %{customdata[1]}"
                                  "<br>%{y:.3f} bits/caractere<extra></extra>")
    fig.add_annotation(x=xs[k], y=ys[k], text="melhor corte", showarrow=True, arrowcolor=TEXTO2, font=dict(color=TEXTO), ay=-36)
    fig.update_xaxes(title="onde cortar (% da profundidade)", ticksuffix="%")
    fig.update_yaxes(title="bits por caractere (menor = melhor)")
    fig.update_layout(showlegend=False)
    return _layout(fig, "Onde costurar: teste rápido de cada ponto de corte (só a rotação)")


def grafico_resultado(r: Receita):
    res = r.resultados
    assuntos = list(res.get("assuntos", {}))
    fig = go.Figure()
    for i, p in enumerate(r.partes):
        fig.add_bar(name=f"só {nome_curto(p.modelo)}", x=assuntos,
                    y=[res["assuntos"][a].get(f"sozinho_{i + 1}") for a in assuntos],
                    marker=dict(color=COR[i % 3], cornerradius=4),
                    hovertemplate="%{x}: %{y:.3f} bits/caractere<extra>só " + nome_curto(p.modelo) + "</extra>")
    fig.add_bar(name="IA nova (costurada)", x=assuntos, y=[res["assuntos"][a].get("costurada") for a in assuntos],
                marker=dict(color=TEXTO, cornerradius=4, pattern=dict(shape="/", fgcolor=SUPERFICIE)),
                hovertemplate="%{x}: %{y:.3f} bits/caractere<extra>IA nova</extra>")
    fig.update_layout(barmode="group", bargap=0.3, bargroupgap=0.08)
    fig.update_yaxes(title="bits por caractere (menor = melhor)", rangemode="tozero")
    fig = _layout(fig, "Resultado: IA nova contra as originais", 360)
    fig.update_layout(legend=dict(orientation="h", y=-0.18, x=0, xanchor="left"), margin=dict(b=80))
    return fig


def grafico_curva(r: Receita):
    c = r.resultados.get("curva", [])
    fig = go.Figure()
    fig.add_scatter(x=[x["epoca"] for x in c], y=[x["treino_bpc"] for x in c], name="treino", mode="lines+markers",
                    line=dict(color=COR[1], width=2, dash="dot"), marker=dict(size=8))
    fig.add_scatter(x=[x["epoca"] for x in c], y=[x["teste_bpc"] for x in c], name="teste (textos nunca vistos)",
                    mode="lines+markers", line=dict(color=COR[0], width=2), marker=dict(size=8))
    fig.update_xaxes(title="época", dtick=1)
    fig.update_yaxes(title="bits por caractere")
    fig.update_layout(hovermode="x unified")
    return _layout(fig, "O super agente aprendendo", 300)


# ----------------------------------------------------------------------------- ações

OCUPADO = ("⏳ As IAs estão ocupadas com outra tarefa (análise, busca do corte, costura ou conversa). "
           "Espere ela terminar e tente de novo.")


def _em_fundo(fn, q):
    """Roda fn numa thread, uma tarefa por vez (as IAs são compartilhadas)."""
    out = {}

    def t():
        if not nucleo.TRAVA.acquire(blocking=False):
            out["erro"] = OCUPADO
            q.put(None)
            return
        try:
            out["r"] = fn()
        except Exception as e:
            out["erro"] = traceback.format_exc()
            if "OutOfMemoryError" in type(e).__name__ or "out of memory" in str(e).lower():
                nucleo.liberar_cache()
                out["erro"] = ("💾 A memória da placa de vídeo acabou. Os modelos dessa tarefa não cabem juntos na placa.\n"
                               "Tente: modelos menores, menos camadas por especialista (no modo cérebro só as camadas "
                               "usadas vão para a placa), 'Tokens por trecho' menor, ou uma placa com mais memória.\n"
                               "Já liberei a placa; pode tentar de novo.\n\n" + out["erro"][-1500:])
        finally:
            nucleo.TRAVA.release()
        q.put(None)
    threading.Thread(target=t, daemon=True).start()
    return out


def analisar(n, m1, m2, m3, *assuntos_v):
    modelos = [m for m in (m1, m2, m3)[:int(n)] if m]
    assuntos = ler_assuntos(*assuntos_v)
    nada = [gr.update()] * 5
    if not assuntos:
        yield ["⚠️ Escolha pelo menos um assunto com textos."] + nada
        return
    q, log = queue.Queue(), ["Medindo cada IA em cada assunto e desligando uma camada por vez para ver quais importam..."]
    yield ["\n".join(log)] + nada
    out = _em_fundo(lambda: nucleo.analisar(modelos, assuntos, log=q.put), q)
    while (msg := q.get()) is not None:
        log.append(msg)
        yield ["\n".join(log[-6:])] + nada
    if "erro" in out:
        yield [out["erro"]] + nada
        return
    res = out["r"]
    (RAIZ / "ultima_analise.json").write_text(json.dumps(res, ensure_ascii=False), encoding="utf-8")
    linhas = ["### O que o super agente achou"]
    for a in res["assuntos"]:
        melhor = min(res["modelos"], key=lambda nm: res["modelos"][nm]["bpc"][a])
        linhas.append(f"- **{a}**: melhor IA é **{nome_curto(melhor)}** ({res['modelos'][melhor]['bpc'][a]:.2f} bits/caractere)")
    for nm, d in res["modelos"].items():
        partes = []
        for a in res["assuntos"]:
            imp = d["importancia"][a]
            top = sorted(sorted(range(len(imp)), key=lambda l: -imp[l])[:3])
            partes.append(f"{a}: camadas {', '.join(map(str, top))}")
        linhas.append(f"- Melhores partes de **{nome_curto(nm)}**: " + "; ".join(partes))
    falante = min(res["modelos"], key=lambda nm: sum(res["modelos"][nm]["bpc"].values()))
    linhas.append(f"\n💡 Deixe **{nome_curto(falante)}** como a IA que **escreve a resposta** (a última): é a que melhor "
                  "escolhe as palavras nos seus assuntos. Depois, em ② Partes, use *Achar o melhor ponto de corte*.")
    graf = [grafico_habilidade(res)] + [grafico_camadas(res, nm, i) for i, nm in enumerate(res["modelos"])]
    graf += [None] * (4 - len(graf))
    log.append("✅ análise pronta")
    yield ["\n".join(log[-2:]), "\n".join(linhas)] + [gr.update(value=g, visible=g is not None) for g in graf]


def sugerir(n, m1, m2, a1, c1):
    nada = [gr.update()] * 3
    if int(n) != 2:
        yield ["A busca automática funciona com 2 IAs."] + nada
        return
    caminhos = [c.strip().strip('"') for c in (c1 or "").split(";") if c.strip()]
    if not caminhos:
        yield ["Escolha o 1º assunto (aba ①) para o super agente testar os cortes nele."] + nada
        return
    textos = nucleo.fatiar(nucleo.ler_textos(caminhos), random.Random(0))
    q, log = queue.Queue(), ["Testando cada ponto de corte com a rotação de Procrustes (rápido)..."]
    yield ["\n".join(log)] + nada
    out = _em_fundo(lambda: nucleo.sugerir_corte(m1.strip(), m2.strip(), textos, log=q.put), q)
    while (msg := q.get()) is not None:
        log.append(msg)
        yield ["\n".join(log)] + nada
    if "erro" in out:
        yield [out["erro"]] + nada
        return
    r = out["r"]
    log.append(f"✅ Melhor: {nome_curto(m1)} até a camada {r['fim_a']} → {nome_curto(m2)} a partir da camada {r['inicio_b']}.")
    yield ["\n".join(log), gr.update(value=r["fim_a"]), gr.update(value=r["inicio_b"]),
           gr.update(value=grafico_cortes(r["curva"]), visible=True)]


def barra_progresso(p: dict | None, inicio: float) -> str:
    """Barra de progresso do treino em HTML."""
    import time as _t
    if not p:
        return ""
    frac = max(0.0, min(1.0, p.get("fracao", 0)))
    decorrido = _t.time() - inicio
    fase = {"carregando": "Carregando as IAs", "preparando": "Preparando o super agente",
            "treinando": "Treinando o super agente", "medindo": "Medindo nos textos de teste",
            "exportando": "Levando para o LM Studio", "orquestra": "Treinando a orquestra", "pronto": "Pronto!"}.get(p.get("fase"), "")
    detalhe = p.get("texto", "")
    sep = " · " if detalhe else ""
    if p.get("fase") == "treinando":
        detalhe = (f"época {p['epoca']} de {p['epocas']} · trecho {p['passo']} de {p['passos']} · "
                   f"erro no treino {p['bpc_treino']:.3f} bits/caractere")
    restam = p.get("restam_s")
    tempo = f"{decorrido / 60:.1f} min" + (f" · faltam ~{max(restam, 0) / 60:.0f} min" if restam is not None and frac < 1 else "")
    ultimas = p.get("curva") or []
    melhor = min((c["teste_bpc"] for c in ultimas), default=None)
    melhor_txt = f"{sep}melhor no teste até agora: <b>{melhor:.3f}</b> bits/caractere" if melhor is not None else ""
    return (f'<div class="cartao"><div style="display:flex;justify-content:space-between;align-items:baseline">'
            f'<b style="font-size:16px">{fase}</b><span style="font-size:22px;font-weight:700">{100 * frac:.0f}%</span></div>'
            f'<div style="height:10px;background:#2c2c2a;border-radius:999px;margin:10px 0;overflow:hidden">'
            f'<div style="height:100%;width:{100 * frac:.1f}%;background:#3987e5;border-radius:999px;transition:width .4s"></div></div>'
            f'<div style="color:#c3c2b7;font-size:13px">{detalhe}{melhor_txt}</div>'
            f'<div style="color:#8b8a83;font-size:12px;margin-top:4px">{tempo}</div></div>')


def costurar(nome, n, *v):
    import time as _t
    _parar.clear()
    n = int(n)
    modelos, lim = v[:MAX_PARTES], v[MAX_PARTES:MAX_PARTES * 3]
    assuntos = ler_assuntos(*v[MAX_PARTES * 3:MAX_PARTES * 3 + MAX_ASSUNTOS * 2])
    epocas, lr, max_tok = v[-3:]
    nada = [gr.update()] * 3
    try:
        r = Receita(nome.strip() or "ia-nova",
                    [Parte(modelos[i].strip(), int(lim[2 * i]), int(lim[2 * i + 1])) for i in range(n)],
                    assuntos, max_tok=int(max_tok), epocas=int(epocas), lr=float(lr))
    except Exception as e:
        yield [f"Erro na receita: {e}", "", ""] + nada
        return
    pasta = SAIDAS / "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in r.nome)
    q, log, inicio = queue.Queue(), [], _t.time()
    prog = {"fase": "carregando", "fracao": 0.01, "texto": "lendo os textos e carregando as IAs na placa"}
    yield ["", barra_progresso(prog, inicio), ""] + nada
    out = _em_fundo(lambda: nucleo.costurar(r, pasta, log=lambda m: q.put(("log", m)), parar=_parar.is_set,
                                            progresso=lambda p: q.put(("prog", p))), q)
    n_curva = 0
    while (item := q.get()) is not None:
        tipo, conteudo = item
        curva_upd = gr.update()
        if tipo == "log":
            log.append(conteudo)
        else:
            prog = conteudo
            c = prog.get("curva") or []
            if len(c) != n_curva and c:      # nova época medida: atualiza o gráfico de aprendizado ao vivo
                n_curva = len(c)
                vista = Receita(r.nome, r.partes, resultados={"curva": list(c)})   # cópia: não mexe na receita do treino
                curva_upd = gr.update(value=grafico_curva(vista), visible=True)
        yield ["\n".join(log[-60:]), barra_progresso(prog, inicio), "", gr.update(), curva_upd, gr.update()]
    if "erro" in out:
        yield ["\n".join(log + ["", out["erro"]]), barra_progresso(prog, inicio), ""] + nada
        return
    r = Receita.abrir(pasta)
    yield ["\n".join(log[-60:]), barra_progresso({"fase": "pronto", "fracao": 1, "curva": r.resultados.get("curva"),
                                                    "texto": f"IA nova salva em ias_novas/{pasta.name}"}, inicio),
           (pasta / "resultado.md").read_text(encoding="utf-8"),
           gr.update(value=grafico_resultado(r), visible=True), gr.update(value=grafico_curva(r), visible=True),
           gr.update(choices=listar_ias(), value=pasta.name)]


def _tipo(p):
    try:
        return json.loads((p / "receita.json").read_text(encoding="utf-8")).get("tipo", "costura")
    except Exception:
        return "?"


def listar_ias():
    return sorted(p.name for p in SAIDAS.iterdir() if (p / "receita.json").exists() and _tipo(p) == "costura")


def listar_cerebros():
    return sorted(p.name for p in SAIDAS.iterdir() if (p / "receita.json").exists() and _tipo(p) == "cerebro")


def ver_ia(nome_ia):
    if not nome_ia:
        return "", gr.update(visible=False)
    r = Receita.abrir(SAIDAS / nome_ia)
    partes = " → ".join(f"{nome_curto(p.modelo)} [camadas {p.inicio}–{p.fim}]" for p in r.partes)
    return f"**{r.nome}** · {partes} · criada em {r.criado}", gr.update(value=grafico_resultado(r), visible=bool(r.resultados))


def conversar(nome_ia, prompt, n, temp, comparar):
    if not nome_ia:
        return "Escolha uma IA nova (crie uma em ③ Costurar).", "", ""
    if not nucleo.TRAVA.acquire(blocking=False):
        return OCUPADO, "", ""
    try:
        import torch
        ia = IACosturada.abrir(SAIDAS / nome_ia, log=lambda *_: None)
        nova = ia.gerar(prompt, int(n), float(temp))
        orig = []
        if comparar:
            for p in (ia.r.partes[0], ia.r.partes[-1]):
                tok, m = nucleo.carregar_modelo(p.modelo)
                x = tok(prompt, return_tensors="pt").to(next(m.parameters()).device)
                with torch.no_grad():
                    o = m.generate(**x, max_new_tokens=int(n), do_sample=float(temp) > 0,
                                   temperature=max(float(temp), 1e-5), pad_token_id=tok.eos_token_id,
                                   repetition_penalty=nucleo.PENALIDADE,
                                   no_repeat_ngram_size=nucleo.NGRAMA)
                orig.append(nucleo.cortar_parada(
                    tok.decode(o[0][x["input_ids"].shape[1]:], skip_special_tokens=True))[0])
        return nova, (orig[0] if orig else ""), (orig[1] if len(orig) > 1 else "")
    except Exception:
        return traceback.format_exc(), "", ""
    finally:
        nucleo.TRAVA.release()



# ----------------------------------------------------------------------------- modo cérebro

def grafico_cerebro(r: ReceitaCerebro):
    res = r.resultados
    assuntos = list(res.get("assuntos", {}))
    fig = go.Figure()
    fig.add_bar(name="só a base", x=assuntos, y=[res["assuntos"][a]["base"] for a in assuntos],
                marker=dict(color=COR[0], cornerradius=4), hovertemplate="%{x}: %{y:.3f}<extra>só a base</extra>")
    fig.add_bar(name="base + especialistas", x=assuntos, y=[res["assuntos"][a].get("cerebro") for a in assuntos],
                marker=dict(color=TEXTO, cornerradius=4, pattern=dict(shape="/", fgcolor=SUPERFICIE)),
                hovertemplate="%{x}: %{y:.3f}<extra>base + especialistas</extra>")
    fig.update_layout(barmode="group", bargap=0.3, bargroupgap=0.08)
    fig.update_yaxes(title="bits por caractere (menor = melhor)", rangemode="tozero")
    fig = _layout(fig, "A base com e sem os especialistas", 340)
    fig.update_layout(legend=dict(orientation="h", y=-0.2, x=0, xanchor="left"), margin=dict(b=80))
    return fig


def grafico_uso(r: ReceitaCerebro):
    res = r.resultados
    assuntos, nomes = list(res.get("assuntos", {})), res.get("nomes", [])
    fig = go.Figure()
    for i, n in enumerate(nomes):
        fig.add_bar(name=n, x=assuntos, y=[100 * res["assuntos"][a]["uso"][i] for a in assuntos],
                    marker=dict(color=COR[(i + 1) % 3], cornerradius=4),
                    text=[f"{100 * res['assuntos'][a]['uso'][i]:.0f}%" for a in assuntos], textposition="outside",
                    hovertemplate="%{x}: aciona %{y:.0f}% das vezes<extra>" + n + "</extra>")
    fig.update_layout(barmode="group", bargap=0.3, bargroupgap=0.08)
    fig.update_yaxes(title="quanto o portão aciona (%)", range=[0, 110], ticksuffix="%")
    fig = _layout(fig, "Quando a base chama cada especialista", 340)
    fig.update_layout(legend=dict(orientation="h", y=-0.2, x=0, xanchor="left"), margin=dict(b=80))
    return fig


def criar_cerebro_ui(nome, base, camada, *v):
    import time as _t
    _parar.clear()
    esp_v, ass_v = v[:8], v[8:8 + MAX_ASSUNTOS * 2]
    epocas, esparsidade = v[-2:]
    esps = []
    for i in range(2):
        nm, modelo, ini, fim = esp_v[4 * i:4 * i + 4]
        if modelo:
            esps.append(Especialista(modelo.strip(), int(ini), int(fim), (nm or f"especialista {i + 1}").strip()))
    nada = [gr.update()] * 3
    if not esps:
        yield ["Escolha pelo menos um especialista.", "", ""] + nada
        return
    r = ReceitaCerebro(nome.strip() or "cerebro", base.strip(), int(camada), esps, ler_assuntos(*ass_v),
                       epocas=int(epocas), esparsidade=float(esparsidade))
    if not r.assuntos:
        yield ["Escolha os assuntos e textos na aba ①.", "", ""] + nada
        return
    pasta = SAIDAS / "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in r.nome)
    q, log, inicio = queue.Queue(), [], _t.time()
    prog = {"fase": "carregando", "fracao": 0.01, "texto": "carregando a base e os especialistas"}
    yield ["", barra_progresso(prog, inicio), ""] + nada
    out = _em_fundo(lambda: nucleo.criar_cerebro(r, pasta, log=lambda m: q.put(("log", m)), parar=_parar.is_set,
                                                 progresso=lambda p: q.put(("prog", p))), q)
    while (item := q.get()) is not None:
        tipo, conteudo = item
        if tipo == "log":
            log.append(conteudo)
        else:
            prog = conteudo
        yield ["\n".join(log[-60:]), barra_progresso(prog, inicio), ""] + nada
    if "erro" in out:
        yield ["\n".join(log + ["", out["erro"]]), barra_progresso(prog, inicio), ""] + nada
        return
    r = ReceitaCerebro.abrir(pasta)
    yield ["\n".join(log[-60:]), barra_progresso({"fase": "pronto", "fracao": 1, "curva": r.resultados.get("curva"),
                                                    "texto": f"IA nova salva em ias_novas/{pasta.name}"}, inicio),
           (pasta / "resultado.md").read_text(encoding="utf-8"),
           gr.update(value=grafico_cerebro(r), visible=True), gr.update(value=grafico_uso(r), visible=True),
           gr.update(choices=listar_cerebros(), value=pasta.name)]


def conversar_cerebro(nome_ia, prompt, n, temp):
    if not nome_ia:
        return "<i>Escolha uma IA do modo cérebro (crie uma acima).</i>"
    if not nucleo.TRAVA.acquire(blocking=False):
        return OCUPADO
    try:
        ia = IACerebro.abrir(SAIDAS / nome_ia, log=lambda *_: None)
        _, pedacos, portoes = ia.gerar(prompt, int(n), float(temp))
        nomes = [e.nome for e in ia.r.especialistas]
        import html as _h
        legenda = " ".join(f'<span style="background:{COR[(i + 1) % 3]}55;border-bottom:2px solid {COR[(i + 1) % 3]};'
                           f'padding:1px 6px;border-radius:4px">{_h.escape(n)}</span>' for i, n in enumerate(nomes))
        spans = []
        for k, p in enumerate(pedacos):
            gs = [portoes[i][k] for i in range(len(portoes))] if portoes else []
            if gs:
                i = max(range(len(gs)), key=lambda j: gs[j])
                cor, forca = COR[(i + 1) % 3], gs[i]
                alfa = int(min(1.0, max(0.0, (forca - 0.3) / 0.6)) * 200)
                titulo = " · ".join(f"{nomes[j]} {100 * gs[j]:.0f}%" for j in range(len(gs)))
                spans.append(f'<span title="{titulo}" style="background:{cor}{alfa:02x};border-radius:3px">'
                             f'{_h.escape(p)}</span>')
            else:
                spans.append(_h.escape(p))
        return (f'<div class="cartao"><div style="color:#8b8a83;font-size:12px;margin-bottom:8px">Cada pedaço é colorido pelo '
                f'especialista que a base mais acionou para escrevê-lo (mais forte = mais acionado; passe o mouse para ver %). '
                f'{legenda}</div><pre style="white-space:pre-wrap;margin:0;font-size:14px;color:#f0efec">'
                f'{_h.escape(prompt)}{"".join(spans)}</pre></div>')
    except Exception:
        return "<pre>" + traceback.format_exc() + "</pre>"
    finally:
        nucleo.TRAVA.release()


def ver_cerebro(nome_ia):
    if not nome_ia:
        return "", gr.update(visible=False), gr.update(visible=False)
    r = ReceitaCerebro.abrir(SAIDAS / nome_ia)
    ok = bool(r.resultados.get("assuntos"))
    return (relatorio_md := (SAIDAS / nome_ia / "resultado.md").read_text(encoding="utf-8")), \
        gr.update(value=grafico_cerebro(r) if ok else None, visible=ok), gr.update(value=grafico_uso(r) if ok else None, visible=ok)

# ----------------------------------------------------------------------------- orquestra

def listar_orquestras():
    return sorted(p.name for p in SAIDAS.iterdir() if (p / "receita.json").exists() and _tipo(p) == "orquestra")


def ver_orquestra(nome):
    if not nome:
        return ""
    arq = SAIDAS / nome / "resultado.md"
    aval = RAIZ / "resultados" / f"avaliacao_{nome}.md"
    txt = arq.read_text(encoding="utf-8") if arq.exists() else ""
    if aval.exists():
        txt += "\n\n---\n" + aval.read_text(encoding="utf-8")
    return txt


def criar_orquestra_ui(nome, base, solista, epocas_o, trechos_o):
    import time as _t
    import orquestra
    _parar.clear()
    if not base or not solista:
        yield "Escolha a IA base e a segunda IA.", "", ""
        return
    r = orquestra.ReceitaOrquestra((nome or "orquestra").strip(), base, [orquestra.Solista(solista)],
                                   epocas=int(epocas_o), max_trechos=int(trechos_o))
    pasta = SAIDAS / "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in r.nome)
    q, log, inicio = queue.Queue(), [], _t.time()
    prog = {"fase": "orquestra", "fracao": 0.01, "texto": "começando"}
    yield "", barra_progresso(prog, inicio), ""

    def tarefa():
        nucleo.liberar_cache()
        orquestra.criar(r, pasta, log=lambda m: q.put(("log", m)), parar=_parar.is_set,
                        progresso=lambda f, t: q.put(("prog", (f, t))))
    out = _em_fundo(tarefa, q)
    while (item := q.get()) is not None:
        tipo_i, c = item
        if tipo_i == "log":
            log.append(c)
        else:
            prog = {"fase": "orquestra", "fracao": c[0], "texto": c[1]}
        yield "\n".join(log[-60:]), barra_progresso(prog, inicio), ""
    nucleo.liberar_cache()
    if "erro" in out:
        yield "\n".join(log + ["", out["erro"]]), barra_progresso(prog, inicio), "❌ Deu erro: veja os detalhes."
        return
    yield ("\n".join(log[-60:]), barra_progresso({"fase": "pronto", "fracao": 1, "texto": r.nome}, inicio),
           (pasta / "resultado.md").read_text(encoding="utf-8") +
           "\n\nPara medir em tarefas de verdade: `python avaliar.py --ia ias_novas/" + pasta.name + "`")


def conversar_orquestra(nome, prompt, n, comparar):
    if not nome:
        return "<i>Escolha uma orquestra.</i>", ""
    if not nucleo.TRAVA.acquire(blocking=False):
        return OCUPADO, ""
    import html as _h
    import torch
    import orquestra
    ia = None
    try:
        nucleo.liberar_cache()
        ia = orquestra.IAOrquestra.abrir(SAIDAS / nome, log=lambda *_: None)
        paradas = ["\nPergunta:", "\nQuestion:", "\n\n\n"]
        texto = ia.completar(prompt, int(n), paradas)
        # quanto a segunda IA participou de cada palavra (o portão da posição anterior decide a próxima)
        tudo = prompt + texto
        e = ia.tok(tudo, return_offsets_mapping=True, add_special_tokens=False)
        d = ia.preparar(tudo, max_tok=10 ** 9)
        with torch.no_grad():
            ia.logits(d)
        w = torch.softmax(ia.ultimo_portao, -1).cpu()
        nomes = [s.nome or nome_curto(s.modelo) for s in ia.r.solistas]
        spans, ini = [], len(prompt)
        for k, (a, b) in enumerate(e["offset_mapping"]):
            if b <= ini or k == 0:
                continue
            pedaco = tudo[max(a, ini):b]
            ws = w[k - 1]
            sol = 1 - float(ws[0])
            alfa = int(min(1.0, sol / 0.5) * 200)
            titulo = f"base {100 * float(ws[0]):.0f}% · " + " · ".join(f"{nm} {100 * float(ws[j + 1]):.0f}%" for j, nm in enumerate(nomes))
            spans.append(f'<span title="{titulo}" style="background:{COR[1]}{alfa:02x};border-radius:3px">{_h.escape(pedaco)}</span>')
        html = (f'<div class="cartao"><div style="color:#8b8a83;font-size:12px;margin-bottom:8px">Cada pedaço fica mais '
                f'colorido quanto mais a <b>segunda IA</b> ({_h.escape(", ".join(nomes))}) participou dele. Passe o mouse para ver a '
                f'divisão.</div><pre style="white-space:pre-wrap;margin:0;font-size:14px;color:#f0efec">{_h.escape(prompt)}'
                f'{"".join(spans)}</pre></div>')
        so_base = ""
        if comparar:
            x = ia.tok(prompt, return_tensors="pt", add_special_tokens=False).to(nucleo.DEV)
            with torch.no_grad():
                o = ia.base.generate(**x, max_new_tokens=int(n), do_sample=False, pad_token_id=ia.tok.eos_token_id)
            so_base = nucleo.cortar_parada(ia.tok.decode(o[0][x["input_ids"].shape[1]:], skip_special_tokens=True), paradas)[0]
        return html, so_base
    except Exception as ex:
        msg = traceback.format_exc()
        if "out of memory" in str(ex).lower():
            msg = ("💾 A placa de vídeo não tem memória para as duas IAs desta orquestra juntas. "
                   "Rode num servidor (RunPod) ou escolha uma orquestra com modelos menores.\n\n" + msg[-800:])
        return "<pre>" + _h.escape(msg) + "</pre>", ""
    finally:
        del ia
        nucleo.liberar_cache()
        nucleo.TRAVA.release()


# ----------------------------------------------------------------------------- exportar para o LM Studio

def ver_exportacao(nome_ia):
    if not nome_ia:
        return "Escolha uma IA nova."
    try:
        import exportar
        r = Receita.abrir(SAIDAS / nome_ia)
        pode, motivo = exportar.checar_fusao(r)
        feito = SAIDAS / nome_ia / "exportacao.json"
        extra = ""
        if feito.exists():
            extra = "\n\n---\n**Última exportação:**\n\n" + exportar.relatorio_exportacao(json.loads(feito.read_text(encoding="utf-8")))
        if pode:
            return (f"✅ **Dá para fazer a fusão exata** ({motivo}). A mudança de base vira parte dos pesos e sai um "
                    f"modelo comum, em poucos minutos." + extra)
        return (f"ℹ️ **Fusão exata não dá:** {motivo}. Vai pela **destilação**: um modelo normal "
                f"(`{r.partes[-1].modelo}`) aprende a imitar a IA costurada. Demora mais (minutos a horas) e "
                f"fica um pouco abaixo da costurada." + extra)
    except Exception:
        return traceback.format_exc()


def exportar_ui(nome_ia, modo, tipo, epocas_d):
    import time as _t
    _parar.clear()
    if not nome_ia:
        yield "Escolha uma IA nova.", "", ""
        return
    import exportar
    modo = {"Automático": "auto", "Fusão exata": "fusao", "Destilação": "destilacao"}[modo]
    q, log, inicio = queue.Queue(), [], _t.time()
    prog = {"fase": "exportando", "fracao": 0.01, "texto": "começando"}
    yield "", barra_progresso(prog, inicio), ""
    out = _em_fundo(lambda: exportar.exportar(SAIDAS / nome_ia, modo=modo, tipo=tipo, epocas=int(epocas_d),
                                              log=lambda m: q.put(("log", m)), parar=_parar.is_set,
                                              progresso=lambda f, t: q.put(("prog", (f, t)))), q)
    while (item := q.get()) is not None:
        tipo_i, c = item
        if tipo_i == "log":
            log.append(c)
        else:
            prog = {"fase": "exportando", "fracao": c[0], "texto": c[1]}
        yield "\n".join(log[-60:]), barra_progresso(prog, inicio), ""
    if "erro" in out:
        yield "\n".join(log + ["", out["erro"]]), barra_progresso(prog, inicio), "❌ Deu erro: veja os detalhes do andamento."
        return
    res = out["r"]
    como = ("Abra o LM Studio → **My Models** (ou a busca de modelos): ela aparece como **costureiro/" +
            Path(res["gguf"]).stem.split(".")[0] + "**. Se não aparecer, feche e abra o LM Studio."
            if res.get("lmstudio") else f"Arquivo: `{res['gguf']}`")
    yield ("\n".join(log[-60:]), barra_progresso({"fase": "pronto", "fracao": 1, "texto": Path(res["gguf"]).name}, inicio),
           exportar.relatorio_exportacao(res) + "\n\n" + como + "\n\n💡 É um modelo de completar texto (não de chat): "
           "no LM Studio, escreva no formato *Pergunta: … Resposta:* e use temperatura baixa.")


def _pecas_ui(vals):
    import fusao
    pecas = []
    for i in range(MAX_PARTES):
        m, a, b = vals[3 * i:3 * i + 3]
        if m and str(m).strip():
            pecas.append(fusao.Peca(str(m).strip(), int(a), int(b)))
    return pecas


def checar_fusao_ui(*vals):
    import fusao
    pecas = _pecas_ui(vals)
    if len(pecas) < 2:
        return "Escolha pelo menos 2 modelos."
    linhas = []
    for p in pecas:
        try:
            n = nucleo.info_modelo(p.modelo)["camadas"]
            linhas.append(f"- `{p.modelo}`: {n} camadas (usando {p.inicio}–{p.fim})")
        except Exception as e:
            linhas.append(f"- `{p.modelo}`: não consegui ler ({e})")
    pode, motivo = fusao.checar([p.modelo for p in pecas])
    cab = (f"✅ **Dá para fundir:** {motivo}." if pode else
           f"❌ **Fusão exata não dá:** {motivo}. Para esse par, costure na aba ③ e use a **destilação** acima.")
    return cab + "\n\n" + "\n".join(linhas)


def fusao_direta_ui(nome_f, *vals):
    import time as _t
    import fusao
    *campos, textos_f, tipo_f, lms_f = vals
    pecas = _pecas_ui(campos)
    if len(pecas) < 2:
        yield "Escolha pelo menos 2 modelos.", "", ""
        return
    caminhos = [c.strip().strip('"') for c in (textos_f or "").split(";") if c.strip()]
    nome_f = (nome_f or "fusao").strip()
    pasta = FUSOES / "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in nome_f)
    q, log, inicio = queue.Queue(), [], _t.time()
    prog = {"fase": "exportando", "fracao": 0.01, "texto": "lendo os textos de calibração"}
    yield "", barra_progresso(prog, inicio), ""

    def tarefa():
        nucleo.liberar_cache()
        textos = fusao.textos_de(caminhos)
        return fusao.fundir_completo(pecas, pasta, textos, nome_f, log=lambda m: q.put(("log", m)),
                                     progresso=lambda f, t: q.put(("prog", (f, t))),
                                     gguf=None if tipo_f == "não gerar" else tipo_f, lmstudio=bool(lms_f))
    out = _em_fundo(tarefa, q)
    while (item := q.get()) is not None:
        tipo_i, c = item
        if tipo_i == "log":
            log.append(c)
        else:
            prog = {"fase": "exportando", "fracao": c[0], "texto": c[1]}
        yield "\n".join(log[-60:]), barra_progresso(prog, inicio), ""
    if "erro" in out:
        yield "\n".join(log + ["", out["erro"]]), barra_progresso(prog, inicio), "❌ Deu erro: veja os detalhes do andamento."
        return
    yield ("\n".join(log[-60:]), barra_progresso({"fase": "pronto", "fracao": 1, "texto": nome_f}, inicio),
           fusao.relatorio(out["r"]) + f"\n\nPasta: `{pasta}` (modelo do Hugging Face em `hf/`, com ficha README.md)")


# ----------------------------------------------------------------------------- interface

TEMA = gr.themes.Base(primary_hue="blue", neutral_hue="stone").set(
    body_background_fill="#0e0e0d", body_background_fill_dark="#0e0e0d", body_text_color="#f0efec",
    body_text_color_dark="#f0efec", body_text_color_subdued="#c3c2b7", body_text_color_subdued_dark="#c3c2b7",
    block_background_fill="#1a1a19", block_background_fill_dark="#1a1a19",
    block_border_color="#2c2c2a", block_border_color_dark="#2c2c2a",
    block_label_text_color="#c3c2b7", block_label_text_color_dark="#c3c2b7",
    block_title_text_color="#c3c2b7", block_title_text_color_dark="#c3c2b7",
    input_background_fill="#232321", input_background_fill_dark="#232321",
    input_border_color="#33332f", input_border_color_dark="#33332f",
    button_primary_background_fill="#3987e5", button_primary_background_fill_dark="#3987e5",
    button_primary_background_fill_hover="#5a9cec", button_primary_background_fill_hover_dark="#5a9cec",
    button_primary_text_color="#0e0e0d", button_primary_text_color_dark="#0e0e0d",
    button_secondary_background_fill="#232321", button_secondary_background_fill_dark="#232321",
    button_secondary_text_color="#f0efec", button_secondary_text_color_dark="#f0efec",
    border_color_primary="#2c2c2a", border_color_primary_dark="#2c2c2a",
    color_accent="#3987e5", slider_color="#3987e5", slider_color_dark="#3987e5",
)

with gr.Blocks(title="Costureiro") as app:
    gr.HTML('<div id="topo"><h1>🧵 Costureiro</h1><p>Junte as <b>melhores partes</b> de IAs diferentes. '
            'Um <i>super agente tradutor</i> aprende a mudança de base entre elas e nasce uma <b>IA nova</b> '
            'focada no que você quiser.</p></div>')

    with gr.Tabs():
        with gr.Tab("① Escolher e analisar"):
            with gr.Row(equal_height=False):
                with gr.Column(scale=3):
                    gr.HTML('<div class="passo">as IAs</div>')
                    n_partes = gr.Radio([2, 3], value=2, label="Quantas IAs costurar")
                    modelos = [gr.Dropdown(SUGESTOES, value=v, allow_custom_value=True, label=f"IA {i + 1}",
                                           info=["lê o texto", "escreve a resposta (com 3 IAs: pensa no meio)",
                                                 "escreve a resposta"][i], visible=i < 2)
                               for i, v in enumerate([SUGESTOES[0], SUGESTOES[2], SUGESTOES[5]])]
                    gr.HTML('<div class="passo" style="margin-top:14px">o foco da IA nova</div>')
                    assuntos = []
                    padroes = [("programação", str(TEXTOS / "programacao")), ("matemática", str(TEXTOS / "matematica")), ("", "")]
                    for j in range(MAX_ASSUNTOS):
                        with gr.Row():
                            assuntos += [gr.Textbox(value=padroes[j][0], label=f"assunto {j + 1}", scale=1),
                                         gr.Textbox(value=padroes[j][1], label="arquivos ou pastas (separe com ;)", scale=3)]
                with gr.Column(scale=2):
                    gr.HTML('<div class="passo">resumo</div>')
                    cartoes = gr.HTML()
                    gr.HTML("<div style='color:#8b8a83;font-size:13px'>Use um nome do Hugging Face ou uma pasta no PC. "
                            "Com 6 GB de placa, some até ~2–3 bilhões de parâmetros. Textos: .txt, .md, .pdf e código.</div>")
            b_analisar = gr.Button("🔍 Analisar: no que cada IA é boa e quais são as melhores partes", variant="primary", size="lg")
            log_an = gr.Textbox(label="Andamento", lines=3, max_lines=6)
            recomendacao = gr.Markdown(elem_classes="recomenda")
            g_hab = gr.Plot(visible=False, show_label=False)
            g_cam = [gr.Plot(visible=False, show_label=False) for _ in range(3)]

        with gr.Tab("② Partes"):
            desenho = gr.HTML()
            gr.HTML('<div class="passo" style="margin-top:12px">quais camadas de cada IA</div>'
                    '<div style="color:#8b8a83;font-size:13px;margin-bottom:6px">Começo = lê as palavras · meio = pensa · '
                    'fim = escolhe as palavras. Use os mapas da análise para escolher as melhores partes.</div>')
            limites = []
            for i in range(MAX_PARTES):
                with gr.Row():
                    limites += [gr.Slider(0, 40, step=1, value=0, label=f"IA {i + 1}: da camada"),
                                gr.Slider(0, 40, step=1, value=0, label="até a camada")]
            b_sugerir = gr.Button("🧠 Super agente: achar o melhor ponto de corte", variant="primary")
            log_sug = gr.Textbox(label="Busca", lines=4, max_lines=8)
            g_cortes = gr.Plot(visible=False, show_label=False)

        with gr.Tab("③ Costurar"):
            with gr.Row():
                nome = gr.Textbox(label="Nome da IA nova", value="codigo-mais-matematica", scale=2)
                epocas = gr.Slider(1, 12, value=6, step=1, label="Épocas")
                lr = gr.Dropdown([1e-4, 3e-4, 1e-3], value=3e-4, label="Velocidade de aprendizado")
                max_tok = gr.Slider(64, 384, value=192, step=32, label="Tokens por trecho")
            with gr.Row():
                b_costurar = gr.Button("🧵 Costurar", variant="primary", size="lg", scale=4)
                b_parar = gr.Button("Parar", size="lg", scale=1)
            barra = gr.HTML()
            with gr.Accordion("Detalhes do andamento", open=False):
                log = gr.Textbox(label="Andamento", lines=10, max_lines=20, autoscroll=True, show_label=False)
            with gr.Row():
                g_res = gr.Plot(visible=False, show_label=False)
                g_curva = gr.Plot(visible=False, show_label=False)
            resultado = gr.Markdown()

        with gr.Tab("🧠 Modo cérebro"):
            gr.HTML('<div class="cartao" style="margin-bottom:10px"><b>Especialistas sob demanda.</b> Uma IA <b>base</b> conversa e '
                    'interpreta. Numa camada dela, um <i>portão</i> decide, palavra por palavra, se precisa chamar um '
                    '<b>especialista</b> — um bloco de camadas de outra IA (ex.: as camadas de código do Coder). '
                    'A pergunta vai traduzida para o especialista, a resposta volta traduzida e é somada ao pensamento da base. '
                    'Uma pequena multa faz o portão só acionar quando ajuda.</div>')
            with gr.Row():
                c_nome = gr.Textbox(label="Nome da IA nova", value="cerebro-codigo-matematica", scale=2)
                c_base = gr.Dropdown(SUGESTOES, value="Qwen/Qwen2.5-0.5B-Instruct", allow_custom_value=True,
                                     label="IA base (conversa e interpreta)", scale=3)
                c_camada = gr.Slider(1, 40, value=12, step=1, label="Camada da base que consulta")
            c_esp = []
            padroes_esp = [("código", "Qwen/Qwen2.5-Coder-0.5B-Instruct", 6, 17),
                           ("matemática", "Qwen/Qwen2.5-Math-1.5B-Instruct", 8, 21)]
            for i, (n_, m_, a_, b_) in enumerate(padroes_esp):
                with gr.Row():
                    c_esp += [gr.Textbox(label=f"especialista {i + 1}", value=n_, scale=1),
                              gr.Dropdown(SUGESTOES, value=m_, allow_custom_value=True, label="IA", scale=3),
                              gr.Number(label="da camada", value=a_, precision=0, scale=1),
                              gr.Number(label="até a camada", value=b_, precision=0, scale=1)]
            gr.HTML("<div style='color:#8b8a83;font-size:13px'>Dica: use a análise da aba ① para escolher as camadas mais "
                    "importantes de cada especialista no assunto dele. Os assuntos e textos também vêm da aba ①. "
                    "Deixe a IA do especialista 2 vazia para usar só um.</div>")
            with gr.Row():
                c_epocas = gr.Slider(1, 10, value=4, step=1, label="Épocas")
                c_esparsidade = gr.Slider(0, 0.1, value=0.02, step=0.005, label="Multa por acionar à toa")
            with gr.Row():
                c_btn = gr.Button("🧠 Criar IA com especialistas", variant="primary", size="lg", scale=4)
                c_parar = gr.Button("Parar", size="lg", scale=1)
            c_barra = gr.HTML()
            with gr.Accordion("Detalhes do andamento", open=False):
                c_log = gr.Textbox(lines=10, max_lines=20, show_label=False, autoscroll=True)
            with gr.Row():
                c_g1 = gr.Plot(visible=False, show_label=False)
                c_g2 = gr.Plot(visible=False, show_label=False)
            c_res = gr.Markdown()
            gr.HTML('<div class="passo" style="margin-top:16px">conversar e ver quem foi acionado</div>')
            with gr.Row():
                c_escolha = gr.Dropdown(listar_cerebros(), label="IA do modo cérebro", scale=5)
                c_atual = gr.Button("↻ atualizar", scale=1)
            c_prompt = gr.Textbox(label="Escreva o começo — a IA continua", lines=3,
                                  value="Pergunta: quanto é a derivada de x^3?\nResposta em Python:\n")
            with gr.Row():
                c_n = gr.Slider(10, 300, value=100, step=10, label="Tamanho da resposta")
                c_temp = gr.Slider(0, 1.2, value=0, step=0.1, label="Criatividade")
            c_gerar = gr.Button("Gerar", variant="primary")
            c_saida = gr.HTML()

        with gr.Tab("🎼 Orquestra"):
            gr.HTML('<div class="cartao" style="margin-bottom:10px"><b>Duas IAs tocando juntas.</b> A IA <b>base</b> roda '
                    'inteira. A <b>segunda IA</b> lê o mesmo texto em paralelo até uma camada funda; uma ponte traduz o '
                    'pensamento dela e um <i>portão</i> decide, palavra por palavra, quanto usar de cada uma. Começa igual à '
                    'base, então não começa pior. O treino imita, em cada palavra, a IA original que acerta melhor.<br><br>'
                    '<b>O que medimos (HumanEval, 164 questões):</b> Coder-1.5B sozinho 46,3% → orquestra Coder + Math '
                    '<b>61,0%</b>. Mas o <b>controle</b> Coder + Coder chega a 58,5%: quase todo o ganho vem do treino da '
                    'ponte (um ajuste fino barato), não da soma das habilidades. Detalhes no README.<br>'
                    '<span style="color:#8b8a83">Treinar uma orquestra de 1.5B precisa de ~12 GB de placa (use um servidor). '
                    'Conversar com uma pronta precisa de ~6 GB.</span></div>')
            gr.HTML('<div class="passo">conversar e ver quando a segunda IA entra</div>')
            with gr.Row():
                o_escolha = gr.Dropdown(listar_orquestras(), label="Orquestra", scale=5)
                o_atual = gr.Button("↻ atualizar", scale=1)
            o_info = gr.Markdown()
            o_prompt = gr.Textbox(label="Escreva o começo — a IA continua", lines=4,
                                  value='def media(numeros):\n    """Devolve a média de uma lista de números."""\n')
            with gr.Row():
                o_n = gr.Slider(10, 400, value=120, step=10, label="Tamanho da resposta")
                o_comparar = gr.Checkbox(value=True, label="Comparar com a base sozinha")
            o_gerar = gr.Button("Gerar", variant="primary", size="lg")
            o_saida = gr.HTML()
            o_base = gr.Textbox(label="só a IA base", lines=8)
            gr.HTML('<div class="passo" style="margin-top:22px">criar uma orquestra</div>')
            with gr.Row():
                o_nome = gr.Textbox(label="Nome", value="orq-coder-com-math", scale=2)
                o_b = gr.Dropdown(SUGESTOES, value="Qwen/Qwen2.5-Coder-1.5B-Instruct", allow_custom_value=True,
                                  label="IA base (escreve a resposta)", scale=3)
                o_s = gr.Dropdown(SUGESTOES, value="Qwen/Qwen2.5-Math-1.5B-Instruct", allow_custom_value=True,
                                  label="segunda IA (lê junto)", scale=3)
            with gr.Row():
                o_ep = gr.Slider(1, 5, value=2, step=1, label="Épocas")
                o_tr = gr.Slider(200, 5000, value=1500, step=100, label="Trechos de treino por assunto")
            gr.HTML("<div style='color:#8b8a83;font-size:13px'>Usa problemas de programação (MBPP, só a parte de treino), "
                    "de matemática (GSM8K, só a parte de treino) e código real. Baixa tudo sozinho. As camadas são "
                    "escolhidas pela rotação. Para um teste justo, crie também um <b>controle</b>: a mesma IA nos dois campos.</div>")
            with gr.Row():
                o_btn = gr.Button("🎼 Criar orquestra", variant="primary", size="lg", scale=4)
                o_parar = gr.Button("Parar", size="lg", scale=1)
            o_barra = gr.HTML()
            with gr.Accordion("Detalhes do andamento", open=False):
                o_log = gr.Textbox(lines=10, max_lines=20, show_label=False, autoscroll=True)
            o_res = gr.Markdown()

        with gr.Tab("④ Conversar"):
            with gr.Row():
                escolha = gr.Dropdown(listar_ias(), label="IA nova", scale=5)
                b_atual = gr.Button("↻ atualizar", scale=1)
            resumo_ia = gr.Markdown()
            g_ia = gr.Plot(visible=False, show_label=False)
            prompt = gr.Textbox(label="Escreva o começo — a IA continua", lines=4,
                                value='def derivada(f, x, h=1e-5):\n    """Derivada numérica de f em x."""\n')
            with gr.Row():
                n_tok = gr.Slider(10, 400, value=120, step=10, label="Tamanho da resposta")
                temp = gr.Slider(0, 1.2, value=0, step=0.1, label="Criatividade (0 = mais provável)")
                comparar = gr.Checkbox(value=True, label="Comparar com as originais")
            b_gerar = gr.Button("Gerar", variant="primary", size="lg")
            saida = gr.Textbox(label="🧵 IA nova", lines=10)
            with gr.Row():
                saida1 = gr.Textbox(label="só a IA que lê", lines=8)
                saida2 = gr.Textbox(label="só a IA que escreve", lines=8)

        with gr.Tab("⑤ Levar para o LM Studio"):
            gr.HTML('<div class="cartao" style="margin-bottom:10px"><b>Transforme a IA costurada num modelo comum (.gguf)</b> '
                    'que abre no LM Studio, Ollama e llama.cpp. <b>Fusão exata</b>: quando as IAs são da mesma família e '
                    'do mesmo tamanho, a mudança de base é embutida nos pesos (rápido). <b>Destilação</b>: para qualquer par, '
                    'um modelo normal aprende a imitar a costurada (mais demorado).</div>')
            with gr.Row():
                e_escolha = gr.Dropdown(listar_ias(), label="IA nova", scale=5)
                e_atual = gr.Button("↻ atualizar", scale=1)
            e_info = gr.Markdown()
            with gr.Row():
                e_modo = gr.Radio(["Automático", "Fusão exata", "Destilação"], value="Automático", label="Como")
                e_tipo = gr.Dropdown(["q8_0", "f16", "bf16"], value="q8_0", label="Formato do .gguf",
                                     info="q8_0 = metade do tamanho, quase sem perda")
                e_epocas = gr.Slider(1, 6, value=3, step=1, label="Épocas da destilação")
            with gr.Row():
                e_btn = gr.Button("📦 Exportar para o LM Studio", variant="primary", size="lg", scale=4)
                e_parar = gr.Button("Parar", size="lg", scale=1)
            e_barra = gr.HTML()
            with gr.Accordion("Detalhes do andamento", open=False):
                e_log = gr.Textbox(lines=10, max_lines=20, show_label=False, autoscroll=True)
            e_res = gr.Markdown()

            gr.HTML('<div class="passo" style="margin-top:22px">fusão direta (sem treinar)</div>'
                    '<div class="cartao" style="margin-bottom:10px">Junte camadas de <b>2 ou 3 modelos da mesma arquitetura e '
                    'do mesmo tamanho</b> (ex.: um Instruct e um Coder do mesmo modelo-base) direto num modelo comum. '
                    'Os pesos são lidos e gravados camada por camada e cada modelo é carregado sozinho, então funciona '
                    'com modelos de qualquer tamanho: em placa de vídeo, em várias placas ou só na memória RAM. '
                    'Serve para Llama, Mistral, Qwen2/2.5, Qwen3 e os MoE (Mixtral, Qwen-MoE).</div>')
            f_campos = []
            padroes_f = [("Qwen/Qwen2.5-3B-Instruct", 0, 19), ("Qwen/Qwen2.5-Coder-3B-Instruct", 20, 35), ("", 0, 0)]
            for i, (m_, a_, b_) in enumerate(padroes_f):
                with gr.Row():
                    f_campos += [gr.Dropdown(SUGESTOES, value=m_ or None, allow_custom_value=True,
                                             label=f"modelo {i + 1}" + (" (opcional)" if i == 2 else ""), scale=4),
                                 gr.Number(label="da camada", value=a_, precision=0, scale=1),
                                 gr.Number(label="até a camada", value=b_, precision=0, scale=1)]
            with gr.Row():
                f_nome = gr.Textbox(label="Nome do modelo novo", value="instruct-coder-3b", scale=2)
                f_textos = gr.Textbox(label="Texto de calibração (arquivos ou pastas, separe com ;)",
                                      value=str(TEXTOS), scale=4)
            with gr.Row():
                f_tipo = gr.Dropdown(["q8_0", "f16", "bf16", "não gerar"], value="q8_0", label="Formato do .gguf")
                f_lms = gr.Checkbox(value=True, label="Copiar para o LM Studio")
            with gr.Row():
                f_checar = gr.Button("🔎 Verificar se combinam", scale=2)
                f_btn = gr.Button("🧬 Fundir", variant="primary", size="lg", scale=4)
            f_info = gr.Markdown()
            f_barra = gr.HTML()
            with gr.Accordion("Detalhes da fusão", open=False):
                f_log = gr.Textbox(lines=10, max_lines=20, show_label=False, autoscroll=True)
            f_res = gr.Markdown()

    c_btn.click(criar_cerebro_ui, [c_nome, c_base, c_camada, *c_esp, *assuntos, c_epocas, c_esparsidade],
                [c_log, c_barra, c_res, c_g1, c_g2, c_escolha])
    c_parar.click(lambda: _parar.set())
    c_atual.click(lambda: gr.update(choices=listar_cerebros()), None, c_escolha)
    c_escolha.input(ver_cerebro, c_escolha, [c_res, c_g1, c_g2])
    c_gerar.click(conversar_cerebro, [c_escolha, c_prompt, c_n, c_temp], c_saida)
    app.load(lambda: gr.update(choices=listar_cerebros()), None, c_escolha)

    def mostrar_ias(n):
        return [gr.update(visible=i < int(n)) for i in range(MAX_PARTES)]

    ent_modelos = [n_partes, *modelos]
    n_partes.change(mostrar_ias, n_partes, modelos)
    for c in ent_modelos:
        c.change(cartoes_modelos, ent_modelos, cartoes)
        c.change(ajustar_limites, ent_modelos, limites)
    for c in [n_partes, *modelos, *limites]:
        c.change(diagrama, [n_partes, *modelos, *limites], desenho)
    app.load(cartoes_modelos, ent_modelos, cartoes)
    app.load(ajustar_limites, ent_modelos, limites)
    app.load(lambda: gr.update(choices=listar_ias()), None, escolha)

    b_analisar.click(analisar, [n_partes, *modelos, *assuntos], [log_an, recomendacao, g_hab, *g_cam])
    b_sugerir.click(sugerir, [n_partes, modelos[0], modelos[1], assuntos[0], assuntos[1]],
                    [log_sug, limites[1], limites[2], g_cortes])
    b_costurar.click(costurar, [nome, n_partes, *modelos, *limites, *assuntos, epocas, lr, max_tok],
                     [log, barra, resultado, g_res, g_curva, escolha])
    b_parar.click(lambda: _parar.set())
    b_atual.click(lambda: gr.update(choices=listar_ias()), None, escolha)
    escolha.input(ver_ia, escolha, [resumo_ia, g_ia])
    b_gerar.click(conversar, [escolha, prompt, n_tok, temp, comparar], [saida, saida1, saida2])

    o_atual.click(lambda: gr.update(choices=listar_orquestras()), None, o_escolha)
    o_escolha.input(ver_orquestra, o_escolha, o_info)
    o_gerar.click(conversar_orquestra, [o_escolha, o_prompt, o_n, o_comparar], [o_saida, o_base])
    o_btn.click(criar_orquestra_ui, [o_nome, o_b, o_s, o_ep, o_tr], [o_log, o_barra, o_res])
    o_parar.click(lambda: _parar.set())
    app.load(lambda: gr.update(choices=listar_orquestras()), None, o_escolha)
    e_atual.click(lambda: gr.update(choices=listar_ias()), None, e_escolha)
    e_escolha.input(ver_exportacao, e_escolha, e_info)
    e_btn.click(exportar_ui, [e_escolha, e_modo, e_tipo, e_epocas], [e_log, e_barra, e_res])
    e_parar.click(lambda: _parar.set())
    f_checar.click(checar_fusao_ui, f_campos, f_info)
    f_btn.click(fusao_direta_ui, [f_nome, *f_campos, f_textos, f_tipo, f_lms], [f_log, f_barra, f_res])
    app.load(lambda: gr.update(choices=listar_ias()), None, e_escolha)

if __name__ == "__main__":
    import os
    # no PC: só abre no próprio computador. Num servidor (RunPod etc.): COSTUREIRO_HOST=0.0.0.0 e/ou COSTUREIRO_LINK=1
    host = os.environ.get("COSTUREIRO_HOST", "127.0.0.1")
    link = os.environ.get("COSTUREIRO_LINK", "") not in ("", "0")
    app.queue().launch(server_name=host, server_port=int(os.environ.get("COSTUREIRO_PORTA", "7861")),
                       inbrowser=host == "127.0.0.1" and not link, share=link, theme=TEMA, css=CSS)
