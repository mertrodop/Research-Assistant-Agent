"""Bounded arithmetic only. No eval, network, imports, names or execution."""
import ast
from fractions import Fraction


def calculate(expression: str) -> str:
    if not expression or len(expression)>160:
        raise ValueError('Arithmetic expression length limit')
    try: tree=ast.parse(expression,mode='eval')
    except SyntaxError as exc: raise ValueError('Invalid arithmetic expression') from exc
    if sum(1 for _ in ast.walk(tree))>35: raise ValueError('Arithmetic expression too complex')
    def visit(n):
        if isinstance(n,ast.Constant) and type(n.value) in (int,float):
            v=Fraction(str(n.value))
        elif isinstance(n,ast.UnaryOp) and isinstance(n.op,(ast.UAdd,ast.USub)):
            v=visit(n.operand); v=-v if isinstance(n.op,ast.USub) else v
        elif isinstance(n,ast.BinOp) and isinstance(n.op,(ast.Add,ast.Sub,ast.Mult,ast.Div,ast.Pow)):
            a,b=visit(n.left),visit(n.right)
            if isinstance(n.op,ast.Add): v=a+b
            elif isinstance(n.op,ast.Sub): v=a-b
            elif isinstance(n.op,ast.Mult): v=a*b
            elif isinstance(n.op,ast.Div):
                if not b: raise ValueError('Division by zero')
                v=a/b
            else:
                if b.denominator!=1 or abs(b)>8 or (not a and b<0): raise ValueError('Exponent limit')
                v=a**int(b)
        else: raise ValueError('Only numeric literals and + - * / ** are allowed')
        if v.numerator.bit_length()>256 or v.denominator.bit_length()>256: raise ValueError('Arithmetic magnitude limit')
        return v
    v=visit(tree.body)
    return str(v.numerator) if v.denominator==1 else f'{v.numerator}/{v.denominator} (approx {float(v):.12g})'
