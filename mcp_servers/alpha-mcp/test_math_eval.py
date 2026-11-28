# Alpha AI (AVA) — testes do avaliador de expressões do alpha-mcp
# Copyright (C) 2026  Felipe lumazini
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.

import math
import unittest

from math_eval import (
    MathExpressionError,
    evaluate,
    format_result,
    normalize,
    safe_eval,
)


class EvaluateTests(unittest.TestCase):
    def test_arithmetic_and_precedence(self):
        cases = {
            "2 + 3 * 4": 14,
            "(2 + 3) * 4": 20,
            "10 / 4": 2.5,
            "10 // 3": 3,
            "10 % 3": 1,
            "-5 + 3": -2,
            "2 ** 8": 256,
        }
        for expression, expected in cases.items():
            with self.subTest(expression=expression):
                self.assertEqual(evaluate(expression), expected)

    def test_notation_normalization(self):
        self.assertEqual(evaluate("2^10"), 1024)
        self.assertEqual(evaluate("5 × 3 ÷ 2"), 7.5)
        self.assertAlmostEqual(evaluate("2 * π"), 2 * math.pi)
        self.assertEqual(normalize("2−1"), "2-1")

    def test_functions_and_constants(self):
        cases = {
            "sqrt(81)": 9,
            "log(100, 10)": 2,
            "log10(1000)": 3,
            "round(3.14159, 2)": 3.14,
            "factorial(10)": 3_628_800,
            "gcd(48, 18)": 6,
            "abs(-7)": 7,
            "max(1, 9, 3)": 9,
            "hypot(3, 4)": 5,
        }
        for expression, expected in cases.items():
            with self.subTest(expression=expression):
                self.assertEqual(evaluate(expression), expected)

    def test_nested_calls(self):
        self.assertEqual(evaluate("sqrt(abs(-16))"), 4)

    def test_trigonometry(self):
        self.assertAlmostEqual(evaluate("sin(pi/2)"), 1.0)
        self.assertAlmostEqual(evaluate("degrees(pi)"), 180.0)


class RejectionTests(unittest.TestCase):
    def test_errors_are_portuguese_and_explicit(self):
        cases = {
            "": "expressão vazia",
            "1/0": "divisão por zero",
            "2 +": "sintaxe",
            "x + 1": "nome desconhecido",
            "__import__('os')": "função não permitida",
            "'texto'": "números reais",
            "1 < 2": "não permitido",
            "a.b": "não permitido",
            "2[0]": "não permitido",
            "2**1001": "expoente muito grande",
            "factorial(1001)": "fatorial muito grande",
            "(-1)**0.5": "complexo",
            "sqrt(-1)": "domínio da função",
        }
        for expression, fragment in cases.items():
            with self.subTest(expression=expression):
                with self.assertRaisesRegex(MathExpressionError, fragment):
                    evaluate(expression)

    def test_safe_eval_never_raises(self):
        for expression in ["", "1/0", "__import__('os')", "x", "2**99999", "2 +"]:
            with self.subTest(expression=expression):
                self.assertTrue(safe_eval(expression).startswith("Erro: "))


class FormatTests(unittest.TestCase):
    def test_result_text(self):
        self.assertEqual(
            format_result("2 + 2", 4),
            "Expressão: 2 + 2\nResultado: 4",
        )

    def test_normalized_line_only_when_changed(self):
        text = safe_eval("2^10")
        self.assertIn("Normalizada: 2**10", text)
        self.assertIn("Resultado: 1024", text)

    def test_floats_without_useless_zeros(self):
        self.assertEqual(safe_eval("10 / 4"), "Expressão: 10 / 4\nResultado: 2.5")


if __name__ == "__main__":
    unittest.main()
