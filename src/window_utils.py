import win32gui
import win32con
import os

def set_window_icon(WINDOW_NAME: str, icon_path: str):
    hwnd = win32gui.FindWindow(None, WINDOW_NAME)
    if hwnd == 0:
        print("Window not found.")
        return

    icon_path = os.path.abspath(icon_path)
    hIcon = win32gui.LoadImage(
        None,
        icon_path,
        win32con.IMAGE_ICON,
        0, 0,
        win32con.LR_LOADFROMFILE | win32con.LR_DEFAULTSIZE
    )

    if hIcon == 0:
        print("Failed to load icon.")
        return

    win32gui.SendMessage(hwnd, win32con.WM_SETICON, win32con.ICON_SMALL, hIcon)
    win32gui.SendMessage(hwnd, win32con.WM_SETICON, win32con.ICON_BIG, hIcon)
