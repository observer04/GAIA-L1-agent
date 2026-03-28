from .postprocess import normalize_answer

__all__ = ["GaiaLangGraphAgent", "build_graph_for_studio", "normalize_answer"]


def __getattr__(name: str):
    if name in {"GaiaLangGraphAgent", "build_graph_for_studio"}:
        from .graph import GaiaLangGraphAgent, build_graph_for_studio

        mapping = {
            "GaiaLangGraphAgent": GaiaLangGraphAgent,
            "build_graph_for_studio": build_graph_for_studio,
        }
        return mapping[name]
    raise AttributeError(f"module 'agent' has no attribute {name!r}")
