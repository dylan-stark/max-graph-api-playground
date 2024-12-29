from max import engine
from max.graph import Graph, TensorType, Type
from max.tensor import Tensor
from max.engine.model import Model

def add_graph() -> Graph:
    graph = Graph(
        in_types=List[Type](
            TensorType(DType.float32, 1),
            TensorType(DType.float32, 1),
        )
    )
    out = graph[0] + graph[1]
    graph.output(out)
    graph.verify()
    return graph

def add_model(graph: Graph) -> Model:
    session = engine.InferenceSession()
    model = session.load(graph)
    return model

def run_add(model: Model, a: Float32, b: Float32) -> Float32:
    input0 = Tensor[DType.float32](List[Float32](a))
    input1 = Tensor[DType.float32](List[Float32](b))
    ret = model.execute("input0", input0^, "input1", input1^)
    return ret.get[DType.float32]("output0")[0]

def main():
    graph = add_graph()
    model = add_model(graph)

    var a: Float32 = 1.0
    var b: Float32 = 1.0
    result = run_add(model, a, b)
    print(a, "+", b, "=", result)
