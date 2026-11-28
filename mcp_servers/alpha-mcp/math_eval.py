# Alpha AI (AVA) — avaliador seguro de expressões matemáticas (alpha-mcp)
# Copyright (C) 2026  Felipe lumazini
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""Avaliador de expressões matemáticas usado pela tool `math_calculate`.

Nunca chama `eval()`: a expressão é parseada com `ast` e só nós explicitamente
permitidos são avaliados. Isso impede execução de código, acesso a atributos,
imports e qualquer efeito colateral, mesmo recebendo texto vindo do modelo.

API pública:
  - `evaluate(expression)` → int|float (levanta `MathExpressionError`).
  - `format_result(expression, value)` → texto exibido ao modelo.
  - `safe_eval(expression)` → texto pronto, inclusive mensagens de erro.
"""

import ast
import math
import operator

MAX_ABS_EXPONENT = 1000     # |base**expoente| enorme travaria o container
MAX_FACTORIAL_ARG = 1000
MAX_NODES = 512
MAX_DEPTH = 64
MAX_INT_BITS = 14_000       # ~4200 casas decimais: abaixo do limite de str(int) do Python

_CONSTANTS = {"pi": math.pi, "e": math.e, "tau": math.tau}

_FUNCTIONS = {
    "abs": abs,
    "round": round,
    "pow": pow,
    "min": min,
    "max": max,
    "floor": math.floor,
    "ceil": math.ceil,
    "trunc": math.trunc,
    "fabs": math.fabs,
    "sqrt": math.sqrt,
    "exp": math.exp,
    "log": math.log,
    "log10": math.log10,
    "log2": math.log2,
    "factorial": math.factorial,
    "gcd": math.gcd,
    "lcm": math.lcm,
    "hypot": math.hypot,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "asin": math.asin,
    "acos": math.acos,
    "atan": math.atan,
    "atan2": math.atan2,
    "sinh": math.sinh,
    "cosh": math.cosh,
    "tanh": math.tanh,
    "degrees": math.degrees,
    "radians": math.radians,
}

_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

# Símbolos que o modelo costuma escrever e que o parser do Python não aceita.
_REPLACEMENTS = (("^", "**"), ("×", "*"), ("÷", "/"), ("π", "pi"), ("−", "-"))

_FUNCTION_HINT = (
    "abs, round, pow, min, max, floor, ceil, sqrt, exp, log, log10, log2, "
    "factorial, gcd, sin, cos, tan, ..."
)


class MathExpressionError(ValueError):
    """Expressão inválida ou não permitida; a mensagem já está em português."""


def normalize(expression: str) -> str:
    """Converte a notação comum de modelos para a sintaxe do Python."""
    text = expression.strip()
    for source, target in _REPLACEMENTS:
        text = text.replace(source, target)
    return text


def _eval_node(node: ast.AST, depth: int) -> int | float:
    if depth > MAX_DEPTH:
        raise MathExpressionError("expressão aninhada demais")

    if isinstance(node, ast.Expression):
        return _eval_node(node.body, depth + 1)

    if isinstance(node, ast.Constant):
        value = node.value
        # bool é subclasse de int e str/bytes/complex não são números reais aqui.
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise MathExpressionError("só são permitidos números reais")
        return value

    if isinstance(node, ast.Name):
        if node.id in _CONSTANTS:
            return _CONSTANTS[node.id]
        raise MathExpressionError(
            f"nome desconhecido: {node.id} (constantes permitidas: pi, e, tau)"
        )

    if isinstance(node, ast.UnaryOp):
        if not isinstance(node.op, (ast.UAdd, ast.USub)):
            raise MathExpressionError("operador unário não permitido")
        value = _eval_node(node.operand, depth + 1)
        return value if isinstance(node.op, ast.UAdd) else -value

    if isinstance(node, ast.BinOp):
        operation = _BINOPS.get(type(node.op))
        if operation is None:
            raise MathExpressionError("operador não permitido (use + - * / // % **)")
        left = _eval_node(node.left, depth + 1)
        right = _eval_node(node.right, depth + 1)
        if isinstance(node.op, ast.Pow) and abs(right) > MAX_ABS_EXPONENT:
            raise MathExpressionError(f"expoente muito grande (limite: {MAX_ABS_EXPONENT})")
        return operation(left, right)

    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in _FUNCTIONS:
            raise MathExpressionError(f"função não permitida (use: {_FUNCTION_HINT})")
        if node.keywords:
            raise MathExpressionError("argumentos nomeados não são permitidos")
        args = [_eval_node(arg, depth + 1) for arg in node.args]
        if node.func.id == "factorial" and args and args[0] > MAX_FACTORIAL_ARG:
            raise MathExpressionError(
                f"fatorial muito grande (limite: {MAX_FACTORIAL_ARG})"
            )
        return _FUNCTIONS[node.func.id](*args)

    raise MathExpressionError(f"construto não permitido: {type(node).__name__}")


def evaluate(expression: str) -> int | float:
    """Avalia a expressão e devolve um número real, ou levanta erro em português."""
    if not isinstance(expression, str) or not expression.strip():
        raise MathExpressionError("expressão vazia")

    text = normalize(expression)
    try:
        tree = ast.parse(text, mode="eval")
    except (SyntaxError, ValueError, MemoryError, RecursionError) as exc:
        raise MathExpressionError("expressão inválida (erro de sintaxe)") from exc

    if sum(1 for _ in ast.walk(tree)) > MAX_NODES:
        raise MathExpressionError("expressão muito longa")

    try:
        value = _eval_node(tree, 0)
    except MathExpressionError:
        raise
    except ZeroDivisionError as exc:
        raise MathExpressionError("divisão por zero") from exc
    except OverflowError as exc:
        raise MathExpressionError("resultado muito grande (overflow)") from exc
    except ValueError as exc:
        if "math domain error" in str(exc):
            raise MathExpressionError(
                "argumento fora do domínio da função (ex.: raiz ou log de número negativo)"
            ) from exc
        raise MathExpressionError(f"valor inválido: {exc}") from exc
    except TypeError as exc:
        raise MathExpressionError(f"valor inválido: {exc}") from exc

    if isinstance(value, complex):
        raise MathExpressionError("resultado complexo — use apenas números reais")
    if isinstance(value, float) and not math.isfinite(value):
        raise MathExpressionError("resultado não finito (inf ou nan)")
    if isinstance(value, int) and value.bit_length() > MAX_INT_BITS:
        raise MathExpressionError("resultado inteiro muito grande")
    return value


def _format_number(value: int | float) -> str:
    if isinstance(value, int):
        return str(value)
    if value.is_integer() and abs(value) < 1e15:
        return str(int(value))
    return f"{value:.12g}"


def format_result(expression: str, value: int | float) -> str:
    """Monta a resposta enviada ao modelo, com a forma normalizada quando muda."""
    text = normalize(expression)
    lines = [f"Expressão: {expression.strip()}"]
    if text != expression.strip():
        lines.append(f"Normalizada: {text}")
    lines.append(f"Resultado: {_format_number(value)}")
    return "\n".join(lines)


def safe_eval(expression: str) -> str:
    """API da tool: devolve o resultado ou uma mensagem `Erro: ...`."""
    try:
        value = evaluate(expression)
    except MathExpressionError as exc:
        return f"Erro: {exc}"
    return format_result(expression, value)
