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
from nucleo import IACosturada, Parte, Receita

RAIZ = Path(__file__).resolve().parent
SAIDAS = RAIZ / "ias_novas"
TEXTOS = RAIZ / "textos"
SAIDAS.mkdir(exist_ok=True)

SUGESTOES = [
    "Qwen/Qwen2.5-Coder-0.5B-Instruct", "Qwen/Qwen2.5-Coder-1.5B-Instruct", "Qwen/Qwen2.5-Math-1.5B-Instruct",
    "Qwen/Qwen2.5-0.5B-Instruct", "Qwen/Qwen2.5-1.5B-Instruct", "Qwen/Qwen3-0.6B", "Qwen/Qwen3-1.7B",
    "HuggingFaceTB/SmolLM2-360M-Instruct", "HuggingFaceTB/SmolLM2-1.7B-Instruct",
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
    return _layout(fig, "Resultado: IA nova contra as originais")


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

def _em_fundo(fn, q):
    out = {}

    def t():
        try:
            out["r"] = fn()
        except Exception:
            out["erro"] = traceback.format_exc()
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


def costurar(nome, n, *v):
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
        yield [f"Erro na receita: {e}", ""] + nada
        return
    pasta = SAIDAS / "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in r.nome)
    q, log = queue.Queue(), []
    out = _em_fundo(lambda: nucleo.costurar(r, pasta, log=q.put, parar=_parar.is_set), q)
    while (msg := q.get()) is not None:
        log.append(msg)
        yield ["\n".join(log[-60:]), ""] + nada
    if "erro" in out:
        yield ["\n".join(log + ["", out["erro"]]), ""] + nada
        return
    yield ["\n".join(log[-60:]), (pasta / "resultado.md").read_text(encoding="utf-8"),
           gr.update(value=grafico_resultado(r), visible=True), gr.update(value=grafico_curva(r), visible=True),
           gr.update(choices=listar_ias(), value=pasta.name)]


def listar_ias():
    return sorted(p.name for p in SAIDAS.iterdir() if (p / "receita.json").exists())


def ver_ia(nome_ia):
    if not nome_ia:
        return "", gr.update(visible=False)
    r = Receita.abrir(SAIDAS / nome_ia)
    partes = " → ".join(f"{nome_curto(p.modelo)} [camadas {p.inicio}–{p.fim}]" for p in r.partes)
    return f"**{r.nome}** · {partes} · criada em {r.criado}", gr.update(value=grafico_resultado(r), visible=bool(r.resultados))


def conversar(nome_ia, prompt, n, temp, comparar):
    if not nome_ia:
        return "Escolha uma IA nova (crie uma em ③ Costurar).", "", ""
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
                                   temperature=max(float(temp), 1e-5), pad_token_id=tok.eos_token_id)
                orig.append(tok.decode(o[0][x["input_ids"].shape[1]:], skip_special_tokens=True))
        return nova, (orig[0] if orig else ""), (orig[1] if len(orig) > 1 else "")
    except Exception:
        return traceback.format_exc(), "", ""


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
            log = gr.Textbox(label="Andamento", lines=10, max_lines=20, autoscroll=True)
            with gr.Row():
                g_res = gr.Plot(visible=False, show_label=False)
                g_curva = gr.Plot(visible=False, show_label=False)
            resultado = gr.Markdown()

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
                     [log, resultado, g_res, g_curva, escolha])
    b_parar.click(lambda: _parar.set())
    b_atual.click(lambda: gr.update(choices=listar_ias()), None, escolha)
    escolha.change(ver_ia, escolha, [resumo_ia, g_ia])
    b_gerar.click(conversar, [escolha, prompt, n_tok, temp, comparar], [saida, saida1, saida2])

if __name__ == "__main__":
    app.queue().launch(server_name="127.0.0.1", server_port=7861, inbrowser=True, theme=TEMA, css=CSS)
