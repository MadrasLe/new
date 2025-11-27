def slim_history(hist, keep=12):
    """Mantém histórico curto pra evitar overflow."""
    if len(hist) > keep:
        return [hist[0]] + hist[-(keep-1):]
    return hist
