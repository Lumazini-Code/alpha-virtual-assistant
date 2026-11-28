---
name: math-calculator
description: Use quando o usuário pedir para calcular, somar, subtrair, multiplicar, dividir, resolver uma conta, calcular porcentagem, juros, raiz, potência, logaritmo, trigonometria ou fatorial, ou quando precisar de um resultado numérico exato em vez de estimar. Ensina a usar a tool math_calculate do alpha-mcp e a escrever expressões matemáticas na sintaxe correta (operadores, funções, constantes, limites e erros comuns).
---

# Calculadora matemática com o MCP `alpha-mcp`

O `alpha-mcp` expõe a tool **`math_calculate(expression)`**: o modelo passa uma expressão e
recebe o resultado. A avaliação é feita por um parser AST com allowlist — **nunca `eval()`** —
então a tool aceita só matemática pura: números reais, operadores e as funções listadas abaixo.

## Quando usar

Use `math_calculate` sempre que o resultado precisar ser exato ou a conta tiver mais de um
passo: porcentagens, juros compostos, conversões, raízes, logaritmos, trigonometria, fatorial,
equações já isoladas. Para `2 + 2` você pode responder direto; para qualquer coisa maior,
chame a tool — não faça conta mental longa.

**Como chamar:**
- Ferramenta direta: `math_calculate` com o argumento `expression`.
- Via Code Mode: `return await tools["alpha-mcp__math_calculate"]({ expression: "2^10" })`.

## Regra de ouro da expressão

Passe **somente a expressão**. Sem pergunta, sem `=`, sem unidades, sem texto, sem variáveis.

| Ruim | Certo |
|---|---|
| `"Quanto é 2+2?"` | `"2+2"` |
| `"x = 2+2"` | `"2+2"` |
| `"1500 + 15%"` | `"1500 * (1 + 0.15)"` |
| `"2(3+4)"` | `"2*(3+4)"` (multiplicação implícita não existe) |
| `"raiz de 9"` | `"sqrt(9)"` |
| `"1.000,50"` | `"1000.50"` (ponto decimal, sem separador de milhar) |

## Operadores e notação

| O que você quer escrever | Use | Exemplo |
|---|---|---|
| Soma, subtração | `+`, `-` | `15 - 3` |
| Multiplicação, divisão | `*`, `/` | `12 * 4`, `10 / 4` |
| Divisão inteira, resto | `//`, `%` | `10 // 3` → `3`, `10 % 3` → `1` |
| Potência | `**` ou `^` | `2**10` ou `2^10` → `1024` |
| Parênteses | `( )` | `(2+3)*4` → `20` |
| Sinal | `-x`, `+x` | `-5 + 3` → `-2` |

A tool normaliza automaticamente `^` → `**`, `×` → `*`, `÷` → `/`, `π` → `pi` e o sinal de
menos tipográfico `−` → `-`. A resposta inclui a linha `Normalizada:` quando houve conversão.

**Precedência** (da maior para a menor): parênteses → potência → sinal → multiplicação/divisão
→ soma/subtração. `-2**2` significa `-(2**2)` = `-4`; use `(-2)**2` para `4`.

## Constantes e funções

Só estes nomes existem (todos em minúsculas, com parênteses):

- **Constantes:** `pi`, `e`, `tau` (equivalente a `2*pi`).
- **Básicas:** `abs(x)`, `round(x, n?)`, `pow(x, y)`, `min(...)`, `max(...)`, `floor(x)`,
  `ceil(x)`, `trunc(x)`, `fabs(x)`.
- **Potência/raiz/log:** `sqrt(x)`, `exp(x)`, `log(x, base?)`, `log10(x)`, `log2(x)`,
  `hypot(x, y)`.
- **Inteiros:** `factorial(n)`, `gcd(a, b)`, `lcm(a, b)`.
- **Trigonometria:** `sin`, `cos`, `tan`, `asin`, `acos`, `atan`, `atan2(y, x)`, `sinh`,
  `cosh`, `tanh`, `degrees` (radianos → graus), `radians` (graus → radianos).

`log(x)` é logaritmo natural; `log(x, 10)` é o decimal. `round(x)` arredonda para inteiro,
`round(x, 2)` para duas casas.

## Exemplos prontos

| Pedido | `expression` | Resultado |
|---|---|---|
| Conta simples | `2 + 3 * 4` | `14` |
| Potência | `2^10` | `1024` |
| 10% de 1500 | `1500 * 0.10` | `150` |
| Juros compostos (10% ao ano, 3 anos) | `1500 * (1 + 0.10) ** 3` | `1996.5` |
| Raiz quadrada | `sqrt(81)` | `9` |
| Log decimal | `log(100, 10)` | `2` |
| Seno de 90° | `sin(pi/2)` | `1` |
| Pi com 4 casas | `round(pi, 4)` | `3.1416` |
| Fatorial | `factorial(10)` | `3628800` |
| MDC | `gcd(48, 18)` | `6` |

## Erros e como corrigir

A tool devolve `Erro: ...` em português. Corrija a expressão e tente de novo:

| Erro | Causa | Correção |
|---|---|---|
| `expressão vazia` | string vazia | envie a conta |
| `divisão por zero` | denominador zero | revise os valores |
| `expressão inválida (erro de sintaxe)` | parênteses, operador ou vírgula fora do lugar | reescreva com `( )` e `.` decimal |
| `nome desconhecido` | variável (`x`, `y`) ou nome inventado | use números e as funções da lista |
| `função não permitida` | função de fora da allowlist | troque pelo equivalente da lista |
| `argumento fora do domínio da função` | `sqrt(-1)`, `log(0)` | o resultado não é real; reformule |
| `resultado complexo` | ex.: `(-1)**0.5` | use apenas números reais |
| `expoente muito grande` / `fatorial muito grande` | limite de 1000 | reduza o valor ou escolha outra forma |

## Limites

- Expoente absoluto máximo: `1000`; fatorial máximo: `1000`.
- Expressões muito longas ou aninhadas são recusadas (limite de nós e profundidade).
- Não há variáveis, atribuição, comparação, strings, listas, indexing nem acesso a atributos.
- Números complexos e resultados infinitos/não finitos não são aceitos.
