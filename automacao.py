import time

try:
    import pyautogui
except Exception:
    pyautogui = None


DEFAULT_MAPPING = {
    "thumbs_up": {"type": "key", "action": "space"},
    "stop": {"type": "key", "action": "s"},
    "left": {"type": "key", "action": "left"},
    "right": {"type": "key", "action": "right"},
}


def perform_action(gesture: str, mapping: dict = None):
    if mapping is None:
        mapping = DEFAULT_MAPPING

    if gesture not in mapping:
        return False

    action = mapping[gesture]
    if pyautogui is None:
        return False

    t = action.get("type")
    if t == "key":
        key = action.get("action")
        pyautogui.press(key)
    elif t == "hotkey":
        keys = action.get("action")
        if isinstance(keys, (list, tuple)):
            pyautogui.hotkey(*keys)
    elif t == "mouse":
        cmd = action.get("action")
        if cmd == "click":
            pyautogui.click()
    time.sleep(0.1)
    return True


if __name__ == "__main__":
    print("Exemplo de automação: pressionando espaço em 1s")
    time.sleep(1)
    perform_action("thumbs_up")
