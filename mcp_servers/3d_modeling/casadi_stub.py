"""Stub mínimo do CasADi para o CadQuery 2.4.

O CadQuery importa `casadi` no módulo do solver de constraints de Assembly,
mesmo quando nenhuma constraint é usada. As tools deste servidor não expõem
Assembly/constraints; manter o pacote real custaria ~277 MB por container.
Qualquer atributo devolve um placeholder inerte — se um scripts usar
constraints de Assembly, o resultado não é suportado (fora do escopo MCP).
"""
class _Placeholder:
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        return _Placeholder()

    def __getattr__(self, name):
        return _Placeholder()

    def __add__(self, other):
        return _Placeholder()

    def __sub__(self, other):
        return _Placeholder()

    def __mul__(self, other):
        return _Placeholder()

    def __truediv__(self, other):
        return _Placeholder()

    def __pow__(self, other):
        return _Placeholder()


def __getattr__(name):
    return _Placeholder
