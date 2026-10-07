import onnx
from collections import Counter

m = onnx.load("yoloe-26n-seg.onnx")

print("INPUTS:")
for i in m.graph.input:
    dims = [d.dim_value for d in i.type.tensor_type.shape.dim]
    print(" ", i.name, dims)

print("\nOUTPUTS:")
for o in m.graph.output:
    dims = [d.dim_value for d in o.type.tensor_type.shape.dim]
    print(" ", o.name, dims)

print("\nOPERATOR TYPES (count):")
for op, n in sorted(Counter(n.op_type for n in m.graph.node).items()):
    print(f"  {op}: {n}")

print("\nLAST 30 NODES:")
for n in list(m.graph.node)[-30:]:
    print(f"  {n.op_type:15} {n.name}  -> {list(n.output)}")