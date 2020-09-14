def isInteger(v):
    x = v
    try:
        x += 1
        return True
    except TypeError:
        return False