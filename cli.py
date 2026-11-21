"""
Costureiro pela linha de comando.

Exemplo (código + matemática, o super agente escolhe o corte):
  python cli.py --a Qwen/Qwen2.5-Coder-0.5B-Instruct --b Qwen/Qwen2.5-Math-1.5B-Instruct ^
      --assunto programação=textos/programacao --assunto matemática=textos/matematica --auto --nome codigo-mais-mat
"""
import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import nucleo
from nucleo import Parte, Receita

p = argparse.ArgumentParser()
p.add_argument("--a", required=True, help="IA que lê o texto")
p.add_argument("--b", required=True, help="IA que fala a resposta")
p.add_argument("--fim_a", type=int, help="última camada usada da IA A")
p.add_argument("--inicio_b", type=int, help="primeira camada usada da IA B")
p.add_argument("--auto", action="store_true", help="o super agente escolhe o melhor corte")
p.add_argument("--assunto", action="append", default=[], help="nome=arquivo_ou_pasta[;outro]")
p.add_argument("--epocas", type=int, default=6)
p.add_argument("--lr", type=float, default=3e-4)
p.add_argument("--max_tok", type=int, default=192)
p.add_argument("--max_trechos", type=int, default=500)
p.add_argument("--nome", default="ia-nova")
p.add_argument("--saida", default=str(Path(__file__).parent / "ias_novas"))
p.add_argument("--prompt", default="def area_circulo(r):\n    \"\"\"Área de um círculo de raio r.\"\"\"\n")
a = p.parse_args()

assuntos = {}
for s in a.assunto:
    nome, caminhos = s.split("=", 1)
    assuntos[nome] = [c for c in caminhos.split(";") if c]

na = nucleo.info_modelo(a.a)["camadas"]
nb = nucleo.info_modelo(a.b)["camadas"]
fim_a, inicio_b = a.fim_a, a.inicio_b
if a.auto:
    print("Super agente procurando o melhor ponto de corte...", flush=True)
    textos = nucleo.fatiar(nucleo.ler_textos([c for v in assuntos.values() for c in v]), random.Random(0))
    r = nucleo.sugerir_corte(a.a, a.b, textos, log=lambda m: print(m, flush=True))
    fim_a, inicio_b = r["fim_a"], r["inicio_b"]
    print(f"Escolhido: A camadas 0..{fim_a}, B camadas {inicio_b}..{nb - 1}", flush=True)
fim_a = na // 2 - 1 if fim_a is None else fim_a
inicio_b = nb // 2 if inicio_b is None else inicio_b

rec = Receita(a.nome, [Parte(a.a, 0, fim_a), Parte(a.b, inicio_b, nb - 1)], assuntos,
              max_tok=a.max_tok, epocas=a.epocas, lr=a.lr, max_trechos=a.max_trechos)
pasta = Path(a.saida) / a.nome
ia, res = nucleo.costurar(rec, pasta, log=lambda m: print(m, flush=True))
print("\n" + (pasta / "resultado.md").read_text(encoding="utf-8"))
print("\n--- exemplo da IA nova ---\n" + a.prompt + ia.gerar(a.prompt, 120), flush=True)
