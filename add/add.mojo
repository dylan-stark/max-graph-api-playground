from max.graph import Graph, TensorType, Type

def main():
    graph = Graph(
        in_types=List[Type](
            TensorType(DType.float32, 1),
            TensorType(DType.float32, 1),
        )
    )

    out = graph[0] + graph[1]

    graph.output(out)

    graph.verify()
    
    print(graph)
