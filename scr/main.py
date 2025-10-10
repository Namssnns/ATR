import customtkinter as ctk
import threading
import time
import os
import json
import requests
import io
import subprocess
import win32gui, win32con
import pyautogui
import numpy as np
import ctypes
import simple as scv
import cv2
import matplotlib.pyplot as plt
import webbrowser
import inspect
from scipy.interpolate import interp1d
from pynput.keyboard import Key, Listener
from ahk import AHK
from urllib.parse import urlparse, parse_qs
from typing import Tuple, Type
from odr import ODR
from PIL import Image
from copy import deepcopy
from tkinter import messagebox 


ahk_path = r"C:\Program Files\AutoHotkey\v2\AutoHotkey64.exe"
ahk = None

if os.path.exists(ahk_path):
    ahk = AHK(executable_path=ahk_path)
else:
    messagebox.showerror(
        "AutoHotkey Not Found",
        "THIS CODE CANNOT RUN BECAUSE AHK WAS NOT FOUND AT:\n"
        "C:\\Program Files\\AutoHotkey\\v2\\\n\n"
        "PLEASE REINSTALL AHK AT THIS LOCATION."
    )

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

ODR_INSTANCE = ODR()
ODR_INSTANCE.load()
ODR_INSTANCE.train()


class MonitorInfo(ctypes.Structure):
    _fields_ = [("cbSize", ctypes.c_uint),
                ("rcMonitor", ctypes.wintypes.RECT),
                ("rcWork", ctypes.wintypes.RECT),
                ("dwFlags", ctypes.c_uint)]


user32 = ctypes.windll.user32
MONITOR_DEFAULTTOPRIMARY = 0x1

class Colors:
    GREEN = (127, 255, 170)
    GRAY_BUTTON = (79, 67, 64)
    GRAY_0 = (27, 23, 22)
    GRAY_1 = (31, 26, 26)
    GRAY_2 = (36, 31, 29)
    GRAY_3 = (45, 38, 37)
    DAYS = {
        "HIGH": ((255, 127, 255), (200, 102, 198), (146, 79, 142)),
        "MRGL": ((160, 214, 6), (128, 167, 10), (98, 122, 17)),
        "SLGT": ((64, 198, 255), (56, 155, 198), (50, 114, 142)),
        "TSTM": ((192, 232, 192), (153, 181, 151), (114, 131, 110)),
        "MDT": ((79, 50, 186), (67, 44, 146), (58, 40, 107)),
        "ENH": ((83, 146, 249), (70, 116, 193), (60, 88, 139)),
    }


CONFIG_FILE = "selections.json"

def load_config():
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_config(config):
    try:
        with open(CONFIG_FILE, "w") as f:
            json.dump(config, f, indent=4)
    except Exception as e:
       pass

def smooth_plot(ax, x, y, color, marker='o', label=None, cap=None):
    x = np.array(x)
    y = np.array(y)

    if cap is not None:
        y = np.clip(y, 0, cap)  
    else:
        y = np.clip(y, 0, None)  

    if len(x) >= 4:
        kind = 'cubic'
    elif len(x) == 3:
        kind = 'quadratic'
    elif len(x) == 2:
        kind = 'linear'
    else:
        kind = None

    if kind and len(x) > 1: 
        f = interp1d(x, y, kind=kind)
        x_smooth = np.linspace(x.min(), x.max(), 300)
        y_smooth = f(x_smooth)
        if cap is not None:
            y_smooth = np.clip(y_smooth, 0, cap)
        else:
            y_smooth = np.clip(y_smooth, 0, None)  
        ax.plot(x_smooth, y_smooth, color=color, linewidth=2, label=label)
    else:
        ax.plot(x, y, color=color, linewidth=2, label=label)

    ax.scatter(x, y, color=color, marker=marker, s=40)



def validate_private_server_code(code: str) -> bool:
    if not code or not code.strip():
        return False
    code = code.strip()
    if len(code) == 32 and code.replace('-', '').replace('_', '').isalnum():
        return True
    
    return False

def build_private_server_url(code: str) -> str:
    return f"https://www.roblox.com/games/14170731342/Twisted-MAIN?privateServerLinkCode={code}"


def extract_private_server_code(url: str) -> str | None:
    parsed = urlparse(url)
    params = parse_qs(parsed.query)
    return params.get("privateServerLinkCode", [None])[0]

class ParameterValidator:
    def __init__(self):
        self.conditions = []
    
    def add_condition(self, parameter, operator, value):
        self.conditions.append({
            'parameter': parameter,
            'operator': operator,
            'value': value
        })
    
    def clear_conditions(self):
        self.conditions.clear()
    
    def validate_data(self, data):
        if not self.conditions:
            return False  
            
        for condition in self.conditions:
            param_name = condition['parameter']
            operator = condition['operator']
            expected_value = condition['value']
            
            actual_value = getattr(data, param_name, None)
            
            if actual_value is None:
                return False  
            
            if isinstance(expected_value, str):
                if operator == "=" and actual_value != expected_value:
                    return False
                elif operator in [">=", "<="] and isinstance(actual_value, str):
                    return False  
            else:
                try:
                    actual_num = float(actual_value) if actual_value is not None else 0
                    expected_num = float(expected_value)
                    
                    if operator == ">=" and actual_num < expected_num:
                        return False
                    elif operator == "<=" and actual_num > expected_num:
                        return False
                    elif operator == "=" and actual_num != expected_num:
                        return False
                except (ValueError, TypeError):
                    return False
        
        return True  
    
    def get_conditions_summary(self):
        if not self.conditions:
            return "No conditions set"
        
        summary = []
        for condition in self.conditions:
            summary.append(f"{condition['parameter']} {condition['operator']} {condition['value']}")
        return ", ".join(summary)

def extract_private_server_code(url: str) -> str | None:
    parsed = urlparse(url)
    params = parse_qs(parsed.query)
    return params.get("privateServerLinkCode", [None])[0]

def has_color(image, target_color, tolerance=20):
    if len(image.shape) == 3 and image.shape[2] >= 3:
        target_bgr = np.array([target_color[2], target_color[1], target_color[0]])
        diff = np.abs(image.astype(np.float32) - target_bgr)
        distances = np.sqrt(np.sum(diff ** 2, axis=2))
        mask = distances <= tolerance
        return np.any(mask)
    return False

class Macro:
    PLACE_ID = "14170731342"
    
    class Data:
        FORMAT = {
            "TEMPERATURE": int, "CAPE": int, "0-3KM LAPSE RATES": float,
            "PWAT": float, "SURFACE RH": int, "DEWPOINT": int,
            "3CAPE": int, "3-6KM LAPSE RATES": float, "SRH": int,
            "700-500mb RH": int, "STP": int, "VTP": int, "DAY1": str, 
            "DAY2": str, "DAY3": str
        }
        
        def __init__(self, **data):
            for name, data_type in self.FORMAT.items():
                try:
                    setattr(self, name, data_type(data.get(name, None)))
                except:
                    setattr(self, name, None)

    def get_data(self, img: np.ndarray) -> tuple[Type[Data], dict]:
        data_output = {}

        format_fields = list(self.Data.FORMAT.items())
        format_index = 0

        gray_mask = scv.mask_color(img, Colors.GRAY_3)
        data_contours = scv.find_contours(gray_mask)
        if not data_contours or len(data_contours) == 0:
            return None, None
        
        try:
            data_contour = max(data_contours, key=cv2.contourArea)
        except (ValueError, RuntimeError):
            return None, None
        data_cutout = scv.extract_contour(img, data_contour)
        
        if data_cutout is None or data_cutout.size == 0 or data_cutout.shape[0] == 0 or data_cutout.shape[1] == 0:
            return None, None

        sub_data_masks = (
            scv.mask_color(data_cutout, Colors.GRAY_0) |
            scv.mask_color(data_cutout, Colors.GRAY_1) |
            scv.mask_color(data_cutout, Colors.GRAY_2)
        )
        sub_data_contours = scv.find_contours(sub_data_masks)
        
        if not sub_data_contours:
            return None, None
            
        sub_data_contours = list(filter(
            lambda e: (scv.get_contour_center(e)[0] < (data_cutout.shape[1] * 2 / 3)) or
                    (scv.get_contour_center(e)[1] < (data_cutout.shape[0] / 4)),
            sub_data_contours
        ))
        
        if not sub_data_contours:
            return None, None
            
        if not sub_data_contours or len(sub_data_contours) < 4:
            return None, None

        sub_data_contours.sort(key=cv2.contourArea, reverse=True)

        if len(sub_data_contours) < 4:
            return None, None

        sub_data_contours, composites_contour, days_contours = sub_data_contours[:2], sub_data_contours[2], sub_data_contours[3:]
        sub_data_contours.sort(key=lambda e: scv.get_contour_center(e)[0])
        days_contours.sort(key=lambda e: scv.get_contour_center(e)[0])

        unit_coef_iterator = iter([0.7, 2.5, 2.6, 1.5, 0.7, 0.7, 2.5, 2.6, 3, 0.7])
        for contour in sub_data_contours:
            contour_cutout = scv.extract_contour(data_cutout, contour)
            contour_cutout[np.where(contour_cutout == Colors.GRAY_0)] = 0

            rows_mask = np.zeros(contour_cutout.shape[:2], np.uint8)
            rows_mask[np.where(contour_cutout.max(axis=1) > 0)[0]] = 255
            rows_contours = scv.find_contours(rows_mask)
            rows_contours.sort(key=lambda e: scv.get_contour_center(e)[1])

            for row_contour in rows_contours:
                if format_index >= len(format_fields):
                    break  

                cont_img = scv.extract_contour(contour_cutout, row_contour)
                color_text_mins = np.min(cont_img, axis=2)
                cont_img = cv2.cvtColor(cont_img, cv2.COLOR_BGR2GRAY)
                color_text = cont_img - color_text_mins
                color_text = scv.spread_hist(color_text)
                color_text[np.where(color_text <= 8)] = 0
                color_text = scv.crop_image(color_text, top=False, bottom=False)
                color_text = color_text[:, :-round(next(unit_coef_iterator, 1) * color_text.shape[0])]
                color_text = scv.upscale(color_text, 16)
                color_text[np.where(color_text <= 140)] = 0
                scv.split_characters(color_text)

                data_name, data_type = format_fields[format_index]
                data_output[data_name] = scv.read_number(ODR(), color_text, data_type)
                format_index += 1

        composites = scv.extract_contour(data_cutout, composites_contour)
        composites[np.where(scv.mask_color(composites, Colors.GRAY_0))] = [0]
        composites = cv2.cvtColor(composites, cv2.COLOR_BGR2GRAY)
        composites[np.where(composites < 40)] = 0
        for composite in (composites[:, :composites.shape[1]//2], composites[:, composites.shape[1]//2:]):
            if format_index >= len(format_fields):
                break
            composite = scv.crop_image(composite)
            composite = scv.upscale(composite, 8)
            composite[np.where(composite <= 140)] = 0

            data_name, data_type = format_fields[format_index]
            data_output[data_name] = scv.read_number(ODR(), composite[:, composite.shape[1]//2:], int)
            format_index += 1

        for i, cont in enumerate(days_contours):
            if format_index >= len(format_fields):
                break
            cont_img = scv.extract_contour(data_cutout, cont)
            data_name, data_type = format_fields[format_index]

            for day_type, colors in Colors.DAYS.items():
                if any(scv.has_color(cont_img, c) for c in colors):
                    data_output[data_name] = day_type
                    break
            else:
                data_output[data_name] = None


            format_index += 1

        code_row = img[:60]
        code_row_mask = scv.mask_color(code_row, Colors.GRAY_3)
        code_contours = scv.find_contours(code_row_mask)
        if not code_contours or len(code_contours) == 0:
            return None, None
        try:
            code_contour = min(code_contours, key=lambda e: scv.get_contour_center(e)[0])
        except (ValueError, RuntimeError):
            return None, None
        code_mask = cv2.drawContours(np.zeros(code_row.shape[:2], np.uint8), [code_contour], -1, 255, -1)
        code_trans = scv.crop_image(scv.mask_transparent(code_row, code_mask))

        top_mask = scv.mask_color(cv2.drawContours(img.copy(), [data_contour], -1, (0, 0, 0), -1), Colors.GRAY_0)
        top_contours = scv.find_contours(top_mask)
        if not top_contours or len(top_contours) == 0:
            return None, None
        try:
            top_contour = max(top_contours, key=cv2.contourArea)
        except (ValueError, RuntimeError):
            return None, None
        full_data_contour = cv2.convexHull(np.vstack((top_contour, data_contour)))
        full_data_mask = cv2.drawContours(np.zeros(img.shape[:2], np.uint8), [full_data_contour], -1, 255, -1)
        data_trans = scv.crop_image(scv.mask_transparent(img, full_data_mask))

        return self.Data(**data_output), {"image": data_trans, "thumbnail": code_trans}
    

class DiscordWebhook:
    def __init__(self, webhook_url=None, role_id=None):
        self.webhook_url = webhook_url
        self.role_id = role_id
        
    def set_webhook(self, webhook_url, role_id=None):
        self.webhook_url = webhook_url
        self.role_id = role_id
        
    def is_configured(self):
        return bool(self.webhook_url and self.webhook_url.strip())
    
    def send_notification(self, conditions_summary, data_summary, server_url=None, screenshot=None):
        if not self.is_configured():
            return False
        try:
            embed = {
                "title": "Server found!",
                "description": (
                    "A server has been found that matches your conditions!\n\n"
                    "**SERVER INFO:**"
                ),
                "color": 0x2ECC71, 
                "fields": [
                    {
                        "name": "Conditions Met",
                        "value": f"```yaml\n{conditions_summary}\n```",
                        "inline": False
                    },
                    {
                        "name": "Server Data",
                        "value": f"```ini\n{data_summary}\n```",
                        "inline": False
                    },
                ],
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime()),
                "footer": {
                    "text": "Made by shortykingy._.",
                }
            }

            if server_url:
                embed["fields"].append({
                    "name": "Private Server Link",
                    "value": f"[Click here to join]({server_url})",
                    "inline": False
                })

            if screenshot:
                embed["image"] = {"url": "attachment://thermos.png"}




            payload = {
                "embeds": [embed],
                "username": "Server Notifier",
            }
            if self.role_id and self.role_id.strip():
                payload["content"] = f"<@{self.role_id.strip()}> Server found!"

            files = None
            if screenshot:
                files = {"file": ("thermos.png", screenshot, "image/png")}



            response = requests.post(
                self.webhook_url,
                data={"payload_json": json.dumps(payload)},
                files=files,
                timeout=10
            )

            return response.status_code == 204
        except Exception as e:
            return False
        
    def send_progress_update(self, interval_stats, total_stats, graph_buffer=None):
        if not self.is_configured():
            return False
            
        try:
            embed = {
                "title": " Progress Update",
                "description": f"**Interval  completed**", 
                "color": 0x3498db,
                "fields": [
                    {
                        "name": " This Interval",
                        "value": (
                            f"```yaml\n"
                            f"Servers: {interval_stats['servers']}\n"
                            f"Failures: {interval_stats['failures']}\n" 
                            f"Success Rate: {interval_stats['success_rate']:.1f}%\n"
                            f"Avg Time: {interval_stats['avg_duration']:.1f}s\n"
                            f"```"
                        ),
                        "inline": True
                    },
                    {
                        "name": " Total Progress",
                        "value": (
                            f"```yaml\n"
                            f"Total Servers: {total_stats['total_servers']}\n"
                            f"Total Failures: {total_stats.get('total_failures', 0)}\n" 
                            f"Overall Rate: {total_stats['overall_success_rate']:.1f}%\n"
                            f"Total Time: {total_stats['total_time']/60:.1f}m\n"
                            f"```"
                        ),
                        "inline": True
                    }
                ],
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime()),
                "footer": {
                    "text": "Progress tracking by shortykingy._.",
                }
            }

            if graph_buffer:
                embed["image"] = {"url": "attachment://progress.png"}

            payload = {
                "embeds": [embed],
                "username": "Progress Tracker",
            }

            files = None
            if graph_buffer:
                files = {"file": ("progress.png", graph_buffer, "image/png")}

            response = requests.post(
                self.webhook_url,
                data={"payload_json": json.dumps(payload)},
                files=files,
                timeout=10
            )

            return response.status_code == 204
        except Exception as e:
            return False

class ProgressTracker:
    def __init__(self):
        self.session_start_time = None
        self.interval_data = []
        self.current_interval = {
            'start_time': None,
            'servers': 0,
            'data': 0,
            'durations': []
        }

    def record_failure(self):
        self.current_interval['failures'] += 1
        
    def start_session(self):
        self.session_start_time = time.time()
        self.interval_data = []
        self.start_new_interval()
        
    def start_new_interval(self):
        self.current_interval = {
            'start_time': time.time(),
            'servers': 0,
            'data': 0,
            'durations': [],
            'highest_stp': 0,  
            'highest_vtp': 0,  
            'failures': 0,
        }
        
    def record_server_joined(self):
        self.current_interval['servers'] += 1
        
    def record_data_collected(self, duration):
        self.current_interval['data'] += 1
        self.current_interval['durations'].append(duration)
        
    def finish_interval(self):
        if self.current_interval['start_time']:
            successes = self.current_interval['servers'] - self.current_interval['failures']
            success_rate = (successes / max(1, self.current_interval['servers'])) * 100
            interval_stats = {
                'start_time': self.current_interval['start_time'],
                'end_time': time.time(),
                'servers': self.current_interval['servers'],
                'data': self.current_interval['data'],
                'failures': self.current_interval['failures'],
                'success_rate': max(0, success_rate), 
                'avg_duration': sum(self.current_interval['durations']) / len(self.current_interval['durations']) if self.current_interval['durations'] else 0,
                'highest_stp': self.current_interval.get('highest_stp', 0),
                'highest_vtp': self.current_interval.get('highest_vtp', 0)
            }


            self.interval_data.append(interval_stats)
            self.start_new_interval()
            return interval_stats
        return None
        
    def get_total_stats(self):
        if not self.interval_data:
            return None
            
        total_servers = sum(interval['servers'] for interval in self.interval_data)
        total_data = sum(interval['data'] for interval in self.interval_data)
        total_failures = sum(interval.get('failures', 0) for interval in self.interval_data) 
        
        all_durations = []
        for interval in self.interval_data:
            if interval['data'] > 0 and interval['avg_duration'] > 0:
                all_durations.extend([interval['avg_duration']] * interval['data'])
        
        total_successes = total_servers - total_failures
        overall_success_rate = (total_successes / max(1, total_servers)) * 100
        
        return {
            'total_servers': total_servers,
            'total_data': total_data,
            'total_failures': total_failures,  
            'overall_success_rate': max(0, overall_success_rate),  
            'overall_avg_duration': sum(all_durations) / len(all_durations) if all_durations else 0,
            'total_time': time.time() - self.session_start_time if self.session_start_time else 0,
            'intervals': len(self.interval_data)  
        }

    
        
    def create_progress_graph(self):
        if len(self.interval_data) < 1:
            return None
            
        plt.style.use('dark_background')
        fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(16, 14))
        fig.tight_layout(pad=3.0)  
        fig.suptitle('Progress Tracking - Interval Statistics', fontsize=16, color='white')
        
        interval_nums = list(range(1, len(self.interval_data) + 1))
        servers = [interval['servers'] for interval in self.interval_data]
        success_rates = [interval['success_rate'] for interval in self.interval_data]
        avg_durations = [interval['avg_duration'] for interval in self.interval_data]
        highest_stp = [interval.get('highest_stp', 0) for interval in self.interval_data]
        highest_vtp = [interval.get('highest_vtp', 0) for interval in self.interval_data]
        thermos_failures = [interval.get('failures', 0) for interval in self.interval_data]
        
        smooth_plot(ax1, interval_nums, servers, color='#3498db')
        ax1.set_title('Servers Joined per Interval')
        ax1.set_xlabel('Interval')
        ax1.set_ylabel('Servers')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(bottom=0)  

        smooth_plot(ax2, interval_nums, thermos_failures, color='#e74c3c')
        ax2.set_title('Thermos Failed per Interval')
        ax2.set_xlabel('Interval')
        ax2.set_ylabel('Thermos Failed')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(bottom=0)  

        smooth_plot(ax3, interval_nums, success_rates, color='#e74c3c', cap=100)
        ax3.set_title('Success Rate per Interval')
        ax3.set_xlabel('Interval')
        ax3.set_ylabel('Success Rate (%)')
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(0, 100)

        smooth_plot(ax4, interval_nums, avg_durations, color='#f39c12')
        ax4.set_title('Average Duration per Interval')
        ax4.set_xlabel('Interval')
        ax4.set_ylabel('Duration (s)')
        ax4.grid(True, alpha=0.3)
        ax4.set_ylim(bottom=0)  

        smooth_plot(ax5, interval_nums, highest_stp, color='#9b59b6')
        ax5.set_title('Highest STP per Interval')
        ax5.set_xlabel('Interval')
        ax5.set_ylabel('STP Value')
        ax5.grid(True, alpha=0.3)
        ax5.set_ylim(bottom=0)  

        smooth_plot(ax6, interval_nums, highest_vtp, color="#ea00ff")
        ax6.set_title('Highest VTP per Interval')
        ax6.set_xlabel('Interval')
        ax6.set_ylabel('VTP Value')
        ax6.grid(True, alpha=0.3)
        ax6.set_ylim(bottom=0) 
        
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight', 
                   facecolor='#2c2c2c', edgecolor='none')
        buffer.seek(0)
        plt.close()
        
        return buffer


class RobloxLauncher:
    def __init__(self, ui_callback=None):
        self.ui_callback = ui_callback  
        self._roblox_shortcut = os.path.join(
            os.path.expandvars("%APPDATA%\\Microsoft\\Windows\\Start Menu\\Programs"),
            "Roblox", "Roblox Player.lnk"
        )
        self.hwnd = 0
        self.last_frame = None
        self.queue = []
        self._queue_lock = threading.Lock()
        self.queue_event = threading.Event()
        self.pause_event = threading.Event()
        self.pause_event.set()

        self.current_server_url = None
        
        self.servers_joined = 0
        self.data_collected = 0
        self.is_running = False
        self.session_durations = []
        self.current_server_start_time = None  

        self.validator = ParameterValidator()
        self.discord_webhook = DiscordWebhook()
        
        self.progress_tracker = ProgressTracker()
        self.progress_enabled = False
        self.progress_interval = 300  
        self.progress_timer = None
        self.failed_servers = 0

        self.data_recorded = False
        
        self.session_start_time = None
        self.session_timeout = 90  
        self.get_data_attempts = 0

    def set_progress_tracking(self, enabled, interval_seconds=300):
        self.progress_enabled = enabled
        self.progress_interval = interval_seconds
        
        if enabled:
            self.log(f"Progress tracking enabled - updates every {interval_seconds}s ({interval_seconds/60:.1f}m)")
        else:
            self.log("Progress tracking disabled")
            
    def start_progress_tracking(self):
        if not self.progress_enabled:
            return
            
        self.progress_tracker.start_session()
        self._schedule_progress_update()
        
    def _schedule_progress_update(self):
        if self.progress_enabled and self.is_running:
            self.progress_timer = threading.Timer(self.progress_interval, self._send_progress_update)
            self.progress_timer.daemon = True
            self.progress_timer.start()
            
    def _send_progress_update(self):
        if not self.progress_enabled or not self.is_running:
            return
            
        try:
            interval_stats = self.progress_tracker.finish_interval()
            total_stats = self.progress_tracker.get_total_stats()
            
            if interval_stats and total_stats:
                self.log(f"Sending progress update - Interval #{len(self.progress_tracker.interval_data)}")
                
                graph_buffer = self.progress_tracker.create_progress_graph()
                
                success = self.discord_webhook.send_progress_update(
                    interval_stats, total_stats, graph_buffer
                )
                
                if success:
                    self.log("Progress update sent successfully")
                else:
                    self.log("Failed to send progress update")

                if graph_buffer:
                    graph_buffer.close()

            self._schedule_progress_update()
            
        except Exception as e:
            self.log(f"Error sending progress update: {e}")
            self._schedule_progress_update()

    def stop_progress_tracking(self):
        if self.progress_timer:
            self.progress_timer.cancel()
            self.progress_timer = None

    def set_parameters(self, conditions):
        self.validator.clear_conditions()
        for condition in conditions:
            self.validator.add_condition(
                condition['parameter'],
                condition['operator'], 
                condition['value']
            )
        self.log(f"Updated conditions: {self.validator.get_conditions_summary()}")

    def log(self, message):
        if self.ui_callback:
            self.ui_callback("log", message)

    def update_stats(self):
        if self.ui_callback:
            failures = self.progress_tracker.current_interval['failures']
            successes = self.servers_joined - failures
            success_rate = (successes / max(1, self.servers_joined)) * 100


            avg_time = (
                sum(self.session_durations) / len(self.session_durations)
                if self.session_durations else 0
            )
            self.ui_callback("stats", {
                "servers": self.servers_joined,
                "data": self.data_collected,
                "rate": f"{success_rate:.1f}%",
                "avg_time": f"{avg_time:.1f}s"
            })


    def find_roblox_window(self):
        hwnds = []
        def enum_windows_callback(hwnd, extra):
            if win32gui.IsWindowVisible(hwnd):
                title = win32gui.GetWindowText(hwnd)
                if "Roblox" in title:
                    hwnds.append(hwnd)
        
        win32gui.EnumWindows(enum_windows_callback, None)
        if hwnds:
            self.hwnd = hwnds[0]
            return hwnds[0]
        return None

    def get_frame(self):
        try:
            if not self.hwnd and not self.find_roblox_window():
                return None
            
            client_rect = ctypes.wintypes.RECT()
            user32.GetClientRect(self.hwnd, ctypes.byref(client_rect))
            top_left = ctypes.wintypes.POINT(0, 0)
            bottom_right = ctypes.wintypes.POINT(client_rect.right, client_rect.bottom)
            user32.ClientToScreen(self.hwnd, ctypes.byref(top_left))
            user32.ClientToScreen(self.hwnd, ctypes.byref(bottom_right))
            self.client_screen_rect = (top_left.x, top_left.y, bottom_right.x, bottom_right.y)
            
            width = bottom_right.x - top_left.x
            height = bottom_right.y - top_left.y
            
            if width <= 0 or height <= 0:
                return self.last_frame
            
            screenshot = pyautogui.screenshot(region=(top_left.x, top_left.y, width, height))
            frame = np.array(screenshot)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            self.last_frame = frame
            return frame
            
        except Exception as e:
            self.log(f"Frame capture error: {e}")
            return self.last_frame



    def check_session_timeout(self):
        if self.session_start_time is None:
            return False
        elapsed = time.time() - self.session_start_time
        return elapsed >= self.session_timeout and not self.data_recorded

    def start_roblox(self, private_server_url):
        try:
            if not os.path.isfile(self._roblox_shortcut):
                self.log("Error: Roblox not installed!")
                if self.progress_enabled:
                    self.progress_tracker.record_failure()
                return False

            code = extract_private_server_code(private_server_url)
            if not code:
                self.log("Error: Invalid private server URL!")
                if self.progress_enabled:
                    self.progress_tracker.record_failure()
                    
                return False

            self.log("Starting Roblox...")
            self.servers_joined += 1
            self.current_server_start_time = time.time()  
            if self.progress_enabled:
                self.progress_tracker.record_server_joined()
            self.update_stats()
            
            subprocess.run([
                "taskkill", "/F", "/IM", "RobloxPlayerBeta.exe"
            ], capture_output=True, timeout=5)
            
            arg = f"roblox://placeId=14170731342&linkCode={code}"
            self.current_server_url = f"https://www.roblox.com/games/14170731342/Twisted-MAIN?privateServerLinkCode={code}"
            subprocess.run([
                "powershell.exe", "-Command",
                f"Start-Process '{self._roblox_shortcut}' -ArgumentList '{arg}'"
            ], timeout=10)
            
            time.sleep(3)
            self.find_roblox_window()
            self.make_fullscreen()
            
            self.session_start_time = time.time()
            self.data_recorded = False

            bottomcenter_point = self.create_point((0.5, 0.5))
            pyautogui.moveTo(*bottomcenter_point) 
            ahk.mouse_move(*bottomcenter_point)
            time.sleep(0.05)
            ahk.click(*bottomcenter_point)
                
            return self.wait_for_game_and_collect_data()
            
        except Exception as e:
            self.log(f"Error starting Roblox: {e}")
            if self.progress_enabled:
                self.progress_tracker.record_failure()
            return False

    def wait_for_game_and_collect_data(self):
        self.log("Waiting for game to load...")
        
        for _ in range(240): 
            if not self.is_running:   
                return False
            if self.check_session_timeout():
                self.log("Session timed out")
                return False
                
            frame = self.get_frame()
            if frame is not None:
                if self.is_selecting(frame):
                    self.log("Spawn screen detected, navigating...")
                    self.select_spawn() 
                elif self.is_game_loaded(frame):
                    self.log("Game loaded, collecting data...")
                    self.get_data_attempts = 0
                    return self.collect_data()
            time.sleep(0.25)
        
        self.log("Game failed to load in time")
        if self.progress_enabled:
            self.progress_tracker.record_failure()
        return False
    

    def select_spawn(self):
        prior = self.create_point((0.5, 0.68))
        center_point = self.create_point((0.5, 0.5))
        ahk.mouse_move(*center_point)
        time.sleep(0.05)
        ahk.click(*center_point)
        ahk.mouse_move(*prior)
        time.sleep(0.2)
        ahk.click(*prior)
        time.sleep(0.1)
        spawn = self.create_point((0.58, 0.40))
        ahk.mouse_move(*spawn)
        time.sleep(0.1)
        ahk.click(*spawn)
        time.sleep(0.1)

        return True
    
    def set_discord_webhook(self, webhook_url, role_id=None):
        self.discord_webhook.set_webhook(webhook_url, role_id)
        if self.discord_webhook.is_configured():
            self.log("Discord webhook configured successfully")
        else:
            self.log("Discord webhook disabled")
    
    def is_selecting(self, img: np.ndarray) -> Tuple[bool, bool]:
        if img is None or len(img.shape) != 3:
            return False, False

        H, W, _ = img.shape

        center_size = min(H, W)
        start_col = W // 2 - center_size // 2
        end_col = W // 2 + center_size // 2

        if start_col >= 0 and end_col <= W and center_size > 0:
            rect_cutout = img[:, start_col:end_col]

            if rect_cutout.size > 0 and len(rect_cutout.shape) == 3:
                green_dominant_pixels = np.count_nonzero(np.argmax(rect_cutout, axis=2) == 1)
                total_pixels = rect_cutout.shape[0] * rect_cutout.shape[1]
                loaded_select = (green_dominant_pixels / total_pixels) > 0.5
            else:
                loaded_select = False
        else:
            loaded_select = False


        return loaded_select 
    
    def is_game_loaded(self, img):
        if img is None or len(img.shape) != 3:
            return False
        H, W, _ = img.shape
        if H > 60 and W > 35:
            button_region = img[30:110, W - 35:W]
            
            return has_color(button_region, (79, 67, 64))
        return False

    def collect_data(self):
        try:
            self.get_data_attempts = self.get_data_attempts + 1
            if self.get_data_attempts > 3:
                self.log("Failed to get data three times. stoping script.")
                if self.progress_enabled:
                    self.progress_tracker.record_failure()
                return False

            
            if not self.is_running: 
                return False
            
            if self.get_data_attempts > 1:
                time.sleep(0.5)
            
            time.sleep(0.5)  


            self.click_data_menu()
            time.sleep(1)
            
            frame = self.get_frame()
            if frame is None:
                if self.progress_enabled:
                    self.progress_tracker.record_failure()
                return False
                
            macro = Macro()
            data, images = macro.get_data(frame)
            
            if data is None:
                self.log("No data found. Most likely cuprit is failing to click open data menu. Trying again...")
                return self.collect_data()
                
            data_values = [getattr(data, name) for name in data.FORMAT]
            meaningful_data = any(value is not None for value in data_values)
            
            if not meaningful_data:
                self.log("No meaningful data found")
                return False
            
            self.log("Data successfully collected!")
            self.data_collected += 1
            self.data_recorded = True
            current_stp = getattr(data, 'STP', 0) or 0
            current_vtp = getattr(data, 'VTP', 0) or 0

            if current_stp > self.progress_tracker.current_interval.get('highest_stp', 0):
                self.progress_tracker.current_interval['highest_stp'] = current_stp
                
            if current_vtp > self.progress_tracker.current_interval.get('highest_vtp', 0):
                self.progress_tracker.current_interval['highest_vtp'] = current_vtp
            if self.progress_enabled and self.current_server_start_time:
                server_duration = time.time() - self.current_server_start_time
                self.progress_tracker.record_data_collected(server_duration)
            self.update_stats()

            if self.validator.validate_data(data):
                condition_summary = self.validator.get_conditions_summary()
                data_summary = []
                for condition in self.validator.conditions:
                    param_name = condition['parameter']
                    actual_value = getattr(data, param_name, None)
                    data_summary.append(f"{param_name}={actual_value}")
                
                data_values_str = ", ".join(data_summary)
                self.log(f"CONDITIONS MET! {condition_summary}")
                self.log(f"Server data: {data_values_str}")
                
                thermos_values = []

                for param_name in data.FORMAT.keys():
                    clean_name = param_name.replace(" ", "")
                    thermos_values.append(f"{clean_name}={getattr(data, param_name, None)}")

                thermos_summary = "\n".join(thermos_values)

                thermos_summary = "\n".join(thermos_values)
                screenshot_buffer = io.BytesIO()
                rgb_image = cv2.cvtColor(images["image"], cv2.COLOR_BGR2RGB)
                screenshot_buffer = io.BytesIO()
                Image.fromarray(rgb_image).save(screenshot_buffer, format="PNG")
                screenshot_buffer.seek(0)

                if self.discord_webhook.is_configured():
                    self.log("Sending Discord notification...")
                    self.log(self.current_server_url)
                    self.discord_webhook.send_notification(
                        condition_summary,
                        thermos_summary,                
                        self.current_server_url,       
                        screenshot_buffer               
                    )
                                
                
                self.is_running = False
                return "STOP_SCRIPT"
            else:
                failed_conditions = []
                for condition in self.validator.conditions:
                    param_name = condition['parameter']
                    operator = condition['operator']
                    expected = condition['value']
                    actual = getattr(data, param_name, None)
                    failed_conditions.append(f"{param_name}={actual} (wanted {operator}{expected})")
                
                self.log(f"Conditions not met: {', '.join(failed_conditions)}")
                
            return True
                
        except Exception as e:
            import traceback
            tb_str = traceback.format_exc()
            self.log(f"Error collecting data: {e}")
            self.log(f"Full traceback:\n{tb_str}")
            if self.progress_enabled:
                self.progress_tracker.record_failure()
            return False

    def make_fullscreen(self):
        if self.hwnd:
            try:
                monitor = user32.MonitorFromWindow(self.hwnd, MONITOR_DEFAULTTOPRIMARY)
                
                monitor_info = MonitorInfo()
                monitor_info.cbSize = ctypes.sizeof(MonitorInfo)
                user32.GetMonitorInfoW(monitor, ctypes.byref(monitor_info))
                
                mon_rect = monitor_info.rcMonitor
                screen_width = mon_rect.right - mon_rect.left
                screen_height = mon_rect.bottom - mon_rect.top
                
                FIXED_WIDTH = screen_width // 4
                FIXED_HEIGHT = screen_height // 2
                
                x = mon_rect.left
                y = mon_rect.top
                
                win32gui.MoveWindow(self.hwnd, x, y, FIXED_WIDTH, FIXED_HEIGHT, True)
                
                win32gui.SetWindowPos(
                    self.hwnd,
                    win32con.HWND_TOPMOST, 
                    x, y, FIXED_WIDTH, FIXED_HEIGHT,
                    win32con.SWP_SHOWWINDOW
                )
                
                win32gui.SetForegroundWindow(self.hwnd)

                self.fixed_width = FIXED_WIDTH
                self.fixed_height = FIXED_HEIGHT
                
                self.log(f"Screen: {screen_width}x{screen_height}, Window: {FIXED_WIDTH}x{FIXED_HEIGHT}")
                
            except Exception as e:
                self.log(f"Error setting window size: {e}")
    def click_data_menu(self):
        bottomcenter_point = self.create_point((0.5, 0.5))
        menu_point = self.create_point((0.43, 0.097))
        pyautogui.moveTo(*bottomcenter_point) #idk why but this fixes the mouse going off the screen
        ahk.mouse_move(*bottomcenter_point)
        time.sleep(0.05)
        ahk.click(*bottomcenter_point)

        ahk.mouse_move(*menu_point)
        time.sleep(0.1)
        ahk.click(*menu_point)
        return True
    

    def create_point(self, rel_point: tuple[float, float]) -> tuple[int, int]:
        if not hasattr(self, 'client_screen_rect') or not self.client_screen_rect:
            if not self.hwnd:
                return 0, 0
            self.get_frame()  
        
        x1, y1, x2, y2 = self.client_screen_rect
        width = x2 - x1
        height = y2 - y1
        
        x = int(x1 + rel_point[0] * width)
        y = int(y1 + rel_point[1] * height)
        
        return x, y

    
    def close_roblox(self):
        if self.current_server_start_time is not None:
            server_duration = time.time() - self.current_server_start_time
            self.session_durations.append(server_duration)
            self.log(f"Server session lasted {server_duration:.1f}s")
            self.current_server_start_time = None
            self.update_stats()  
        try:
            if self.hwnd and win32gui.IsWindow(self.hwnd):
                try:
                    
                    center_point = self.create_point((0.5, 0.1))
                    ahk.mouse_move(*center_point)
                    ahk.click()
                    time.sleep(1)
                    ahk.key_press('escape')
                    time.sleep(0.25)
                    ahk.key_press('L')
                    time.sleep(0.25)
                    ahk.key_press('enter') 
                    time.sleep(0.25)
                except:
                    print("broken")
                    pass  
        except:
            pass  
        try:
            subprocess.run([
                "taskkill", "/F", "/IM", "RobloxPlayerBeta.exe"
            ], timeout=3, capture_output=True)
            
            self.hwnd = 0
            return True
        except Exception as e:
            return False


    def main_loop(self, private_server_url):
        self.is_running = True
        retry_count = 0
        

        if self.progress_enabled:
            self.start_progress_tracking()
        
        while self.is_running:
            try:
                self.log(f"Server attempt {retry_count + 1}")
                
                if not self.is_running:
                    self.log("Launcher stopped by user")
                    break
                
                result = self.start_roblox(private_server_url)
                
                if result == "STOP_SCRIPT":
                    self.log("Script stopped - conditions met!") 
                    break
                elif result:
                    self.log("Data collected successfully!")
                    self.close_roblox()
                    time.sleep(1)
                else:
                    if self.is_running:
                        self.log("Failed to collect data, trying new server")
                        self.close_roblox()
                    else:
                        break
                
                retry_count += 1
                
            except Exception as e:
                self.log(f"Loop error: {e}")
                if self.is_running: 
                    self.close_roblox()
                retry_count += 1
                time.sleep(1)
        
        self.is_running = False
        self.stop_progress_tracking()
        self.log("Launcher stopped")

class ParameterSelectorDialog:

    def __init__(self, parent, data_format):
        self.parent = parent
        self.data_format = data_format
        self.conditions = []
        
        self.dialog = ctk.CTkToplevel(parent)
        self.dialog.title("Parameter Selector")
        self.dialog.geometry("700x500")
        self.dialog.transient(parent)
        self.dialog.grab_set()
        
        self.setup_ui()
        
    def setup_ui(self):

        title = ctk.CTkLabel(
            self.dialog,
            text="Set Parameter Conditions",
            font=ctk.CTkFont(size=20, weight="bold")
        )
        title.pack(pady=20)
        
        instructions = ctk.CTkLabel(
            self.dialog,
            text="Add conditions that servers must meet. The script will stop when a server matches ALL conditions.",
            wraplength=500
        )
        instructions.pack(pady=10)
        
        input_frame = ctk.CTkFrame(self.dialog)
        input_frame.pack(pady=20, padx=20, fill="x")
        
        ctk.CTkLabel(input_frame, text="Parameter:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.param_var = ctk.StringVar()
        self.param_dropdown = ctk.CTkComboBox(
            input_frame,
            variable=self.param_var,
            values=list(self.data_format.keys()),
            width=150
        )
        self.param_dropdown.grid(row=0, column=1, padx=5, pady=5)
        
        ctk.CTkLabel(input_frame, text="Operator:").grid(row=0, column=2, padx=5, pady=5, sticky="w")
        self.op_var = ctk.StringVar()
        self.op_dropdown = ctk.CTkComboBox(
            input_frame,
            variable=self.op_var,
            values=[">=", "<=", "="],
            width=80
        )
        self.op_dropdown.grid(row=0, column=3, padx=5, pady=5)
        
        ctk.CTkLabel(input_frame, text="Value:").grid(row=0, column=4, padx=5, pady=5, sticky="w")
        self.value_entry = ctk.CTkEntry(input_frame, width=100)
        self.value_entry.grid(row=0, column=5, padx=5, pady=5)

        add_btn = ctk.CTkButton(
            input_frame,
            text="Add Condition",
            command=self.add_condition,
            width=100
        )
        add_btn.grid(row=0, column=6, padx=10, pady=5)
        
        conditions_frame = ctk.CTkFrame(self.dialog)
        conditions_frame.pack(pady=20, padx=20, fill="both", expand=True)
        
        ctk.CTkLabel(
            conditions_frame,
            text="Current Conditions:",
            font=ctk.CTkFont(size=14, weight="bold")
        ).pack(pady=10)
        
        self.conditions_list = ctk.CTkScrollableFrame(conditions_frame)
        self.conditions_list.pack(fill="both", expand=True, padx=10, pady=10)
        
        button_frame = ctk.CTkFrame(self.dialog)
        button_frame.pack(pady=20, fill="x")
        
        clear_btn = ctk.CTkButton(
            button_frame,
            text="Clear All",
            command=self.clear_conditions,
            fg_color="#dc3545",
            hover_color="#c82333"
        )
        clear_btn.pack(side="left", padx=20)
        
        save_btn = ctk.CTkButton(
            button_frame,
            text="Save & Close",
            command=self.save_and_close,
            fg_color="#28a745",
            hover_color="#218838"
        )
        save_btn.pack(side="right", padx=20)
        
        cancel_btn = ctk.CTkButton(
            button_frame,
            text="Cancel",
            command=self.dialog.destroy
        )
        cancel_btn.pack(side="right", padx=10)
        
    def add_condition(self):
        param = self.param_var.get()
        operator = self.op_var.get()
        value_str = self.value_entry.get().strip()
        
        if not param or not operator or not value_str:
            messagebox.showerror("Error", "Please fill all fields")
            return
        
        for existing_condition in self.conditions:
            if existing_condition['parameter'] == param:
                messagebox.showerror("Error", f"Parameter '{param}' already has a condition set. Remove it first to add a new one.")
                return
            
        param_type = self.data_format.get(param)
        if param_type == str:
            if operator in [">=", "<="]:
                messagebox.showerror("Error", "Cannot use >= or <= with string parameters")
                return
            value = value_str
        else:
            try:
                value = param_type(value_str) if param_type else value_str
            except ValueError:
                messagebox.showerror("Error", f"Invalid value for {param}")
                return
        
        condition = {
            'parameter': param,
            'operator': operator,
            'value': value
        }
        
        self.conditions.append(condition)
        self.update_conditions_display()
        self.value_entry.delete(0, 'end')
        
    def remove_condition(self, index):
        if 0 <= index < len(self.conditions):
            self.conditions.pop(index)
            self.update_conditions_display()
            
    def clear_conditions(self):
        self.conditions.clear()
        self.update_conditions_display()
        
    def update_conditions_display(self):
        for widget in self.conditions_list.winfo_children():
            widget.destroy()
            
        if not self.conditions:
            no_conditions = ctk.CTkLabel(
                self.conditions_list,
                text="No conditions set",
                text_color="gray"
            )
            no_conditions.pack(pady=20)
            return
            
        for i, condition in enumerate(self.conditions):
            condition_frame = ctk.CTkFrame(self.conditions_list)
            condition_frame.pack(fill="x", pady=5, padx=5)
            
            condition_text = f"{condition['parameter']} {condition['operator']} {condition['value']}"
            condition_label = ctk.CTkLabel(
                condition_frame,
                text=condition_text,
                font=ctk.CTkFont(size=12)
            )
            condition_label.pack(side="left", padx=10, pady=5)
            
            remove_btn = ctk.CTkButton(
                condition_frame,
                text="Remove",
                command=lambda idx=i: self.remove_condition(idx),
                width=60,
                height=25,
                fg_color="#dc3545",
                hover_color="#c82333"
            )
            remove_btn.pack(side="right", padx=10, pady=5)
            
    def save_and_close(self):
        self.dialog.destroy()
        
    def get_conditions(self):
        return self.conditions


class UI:

    def __init__(self):
        self.root = ctk.CTk()
        self.root.title("ATR")
        self.root.geometry("700x700")
        self.root.resizable(True, True)
        
        self.root.grid_columnconfigure(1, weight=1)
        self.root.grid_rowconfigure(0, weight=1)
        
        self.launcher = RobloxLauncher(ui_callback=self.handle_launcher_callback)

        self.config = load_config()

        
        self.servers_stat = None
        self.data_stat = None
        self.rate_stat = None

        self.setup_hotkeys()
        self.create_main_content()
        self.restore_config()

    def setup_hotkeys(self):
 
        def on_key_press(key):
            try:
                if key == Key.f2:
                    self.root.after(0, self.start_launcher_hotkey)
                elif key == Key.f3:
                    self.root.after(0, self.stop_launcher_hotkey)
                    self.root.deiconify()
            except AttributeError:
                pass
        
        self.keyboard_listener = Listener(on_press=on_key_press)
        self.keyboard_listener.daemon = True
        self.keyboard_listener.start()
        
        self.log_message("Hotkeys enabled: F2 = Start, F3 = Stop")

    def start_launcher_hotkey(self):
        if self.start_btn.cget("state") == "normal":
            self.log_message("F2 pressed - Starting launcher via hotkey")
            self.start_launcher()
        else:
            self.log_message("F2 pressed - Launcher already running")

    def stop_launcher_hotkey(self):
        if self.stop_btn.cget("state") == "normal":
            self.log_message("F3 pressed - Stopping launcher via hotkey")
            self.stop_launcher()
        else:
            self.log_message("F3 pressed - Launcher not running")



    def restore_config(self):

        if "progress_enabled" in self.config:
            self.progress_enabled_var.set(self.config["progress_enabled"])
            
        if "progress_interval" in self.config:
            self.progress_interval_var.set(str(self.config["progress_interval"]))
    
        self.toggle_progress_options()
        if "webhook_url" in self.config:
            self.webhook_entry.insert(0, self.config["webhook_url"])

        if "user_id" in self.config:  

            self.UserId_entry.insert(0, self.config["user_id"])

        if "private_server_url" in self.config:
            self.url_entry.insert(0, self.config["private_server_url"])

        if "parameters" in self.config:
            self.launcher.set_parameters(self.config["parameters"])
            conds = self.config["parameters"]
            if conds:
                cond_text = "; ".join(
                    f"{c['parameter']} {c['operator']} {c['value']}" for c in conds
                )
                self.conditions_display.configure(text=cond_text)
        if "user_id" in self.config and "webhook_url" in self.config: 
            if self.config["user_id"] == "":
                self.launcher.set_discord_webhook(self.config["webhook_url"])
            else:
                self.launcher.set_discord_webhook(self.config["webhook_url"], self.config["user_id"])
        progress_enabled = self.config.get("progress_enabled", False)
        progress_interval = self.config.get("progress_interval", 300)
        self.launcher.set_progress_tracking(progress_enabled, progress_interval)

        

    def handle_launcher_callback(self, callback_type, data):
        if callback_type == "log":
            self.log_message(data)
        elif callback_type == "stats":
            self.update_statistics(data)
        elif callback_type == "match_found":
            self.on_match_found()


    def create_main_content(self):
        self.main_frame = ctk.CTkFrame(self.root)
        self.main_frame.grid(row=0, column=1, sticky="nsew", padx=20, pady=20)
        self.main_frame.grid_columnconfigure(0, weight=1)
        self.main_frame.grid_rowconfigure(1, weight=1)
        
        self.create_content_widgets()

    def create_content_widgets(self):
        self.content_frame = ctk.CTkScrollableFrame(self.main_frame)
        self.content_frame.grid(row=1, column=0, sticky="nsew", padx=20, pady=20)
        self.content_frame.grid_columnconfigure((0, 1), weight=1)
        
        self.create_control_card()
        self.create_parameter_card()
        self.create_discord_card()
        self.create_statistics_cards()
        self.create_logs_section()
        self.create_credits_section()  

    def create_credits_section(self):
        credits_card = ctk.CTkFrame(self.content_frame)
        credits_card.grid(row=5, column=0, columnspan=2, sticky="ew", pady=20)
        credits_card.grid_columnconfigure(0, weight=1)
        
        title = ctk.CTkLabel(
            credits_card,
            text="Credits",
            font=ctk.CTkFont(size=18, weight="bold")
        )
        title.pack(pady=(20, 10))
        
        credits_text = ctk.CTkLabel(
            credits_card,
            text="Portions of this project were created or based on code from",
            font=ctk.CTkFont(size=12)
        )
        credits_text.pack(pady=(0, 5))
        
        retwisted_link = ctk.CTkButton(
            credits_card,
            text="ReTwisted by @Okmada",
            command=lambda: webbrowser.open("https://github.com/Okmada/ReTwisted"),
            fg_color="transparent",
            text_color=("blue", "lightblue"),
            hover_color=("lightgray", "darkgray"),
            font=ctk.CTkFont(size=12, underline=True),
            height=25
        )
        retwisted_link.pack(pady=(0, 20))


    def create_control_card(self):
        control_card = ctk.CTkFrame(self.content_frame)
        control_card.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 20))
        control_card.grid_columnconfigure(1, weight=1)
        
        title = ctk.CTkLabel(
            control_card,
            text="Launcher Controls",
            font=ctk.CTkFont(size=18, weight="bold")
        )
        title.grid(row=0, column=0, columnspan=3, padx=20, pady=(20, 10), sticky="w")
        
        url_label = ctk.CTkLabel(control_card, text="Private Server URL:")
        url_label.grid(row=1, column=0, padx=20, pady=5, sticky="w")
        
        self.url_entry = ctk.CTkEntry(
            control_card,
            placeholder_text="Enter the 32 character code at the end of the redirct code of a private server link. Ex: 12345678912345678912345678912345",
            width=400,
            height=35
        )
        self.url_entry.grid(row=2, column=0, columnspan=2, padx=20, pady=5, sticky="ew")
        
        default_url = ""
        self.url_entry.insert(0, default_url)
        
        button_frame = ctk.CTkFrame(control_card, fg_color="transparent")
        button_frame.grid(row=3, column=0, columnspan=3, padx=20, pady=20, sticky="ew")
        button_frame.grid_columnconfigure((0, 1), weight=1)
        
        self.start_btn = ctk.CTkButton(
            button_frame,
            text="Start Launcher (F2)",
            height=40,
            font=ctk.CTkFont(size=14, weight="bold"),
            command=self.start_launcher,
            fg_color="#28a745",
            hover_color="#218838"
        )
        self.start_btn.grid(row=0, column=0, padx=10, sticky="ew")
        
        self.stop_btn = ctk.CTkButton(
            button_frame,
            text="Stop Launcher (F3)",
            height=40,
            font=ctk.CTkFont(size=14, weight="bold"),
            command=self.stop_launcher ,
            fg_color="#dc3545",
            hover_color="#c82333",
            state="disabled"
        )
        self.stop_btn.grid(row=0, column=1, padx=10, sticky="ew")

    def create_discord_card(self):
            discord_card = ctk.CTkFrame(self.content_frame)
            discord_card.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(0, 20))
            discord_card.grid_columnconfigure(1, weight=1)
            
            title = ctk.CTkLabel(
                discord_card,
                text="Discord Notifications",
                font=ctk.CTkFont(size=18, weight="bold")
            )
            title.grid(row=0, column=0, padx=20, pady=(20, 10), sticky="w")
            
            desc = ctk.CTkLabel(
                discord_card,
                text="Configure Discord webhook to get notified when a matching server is found. Leave blank to disable.",
                wraplength=400
            )
            desc.grid(row=1, column=0, columnspan=2, padx=20, pady=5, sticky="w")
            
            webhook_label = ctk.CTkLabel(discord_card, text="Discord Webhook URL:")
            webhook_label.grid(row=2, column=0, padx=20, pady=(10, 5), sticky="w")
            
            self.webhook_entry = ctk.CTkEntry(
                discord_card,
                placeholder_text="Enter your webhook here.",
                width=400,
                height=35
            )
            self.webhook_entry.grid(row=3, column=0, columnspan=2, padx=20, pady=5, sticky="ew")
            
            UserId_label = ctk.CTkLabel(discord_card, text="User ID to ping (optional):")
            UserId_label.grid(row=4, column=0, padx=20, pady=(10, 5), sticky="w")
            
            self.UserId_entry = ctk.CTkEntry(
                discord_card,
                placeholder_text="123456789012345678",
                width=200,
                height=35
            )
            self.UserId_entry.grid(row=5, column=0, padx=20, pady=5, sticky="w")
            
            test_btn = ctk.CTkButton(
                discord_card,
                text="Test Webhook",
                command=self.test_webhook,
                height=35,
                width=120
            )
            test_btn.grid(row=5, column=1, padx=20, pady=5, sticky="e")
            
            progress_frame = ctk.CTkFrame(discord_card, fg_color="transparent")
            progress_frame.grid(row=6, column=0, columnspan=2, padx=20, pady=(20, 10), sticky="ew")
            progress_frame.grid_columnconfigure(1, weight=1)
            
            progress_title = ctk.CTkLabel(
                progress_frame,
                text="Progress Tracking",
                font=ctk.CTkFont(size=14, weight="bold")
            )
            progress_title.grid(row=0, column=0, columnspan=3, pady=(0, 10), sticky="w")
            
            self.progress_enabled_var = ctk.BooleanVar()
            self.progress_checkbox = ctk.CTkCheckBox(
                progress_frame,
                text="Send progress graphs every",
                variable=self.progress_enabled_var,
                command=self.toggle_progress_options
            )
            self.progress_checkbox.grid(row=1, column=0, padx=0, pady=5, sticky="w")

            interval_frame = ctk.CTkFrame(progress_frame, fg_color="transparent")
            interval_frame.grid(row=1, column=1, padx=(10, 5), pady=5, sticky="w")
            
            self.progress_interval_var = ctk.StringVar(value="300")
            self.progress_interval_entry =  ctk.CTkEntry(
                interval_frame,
                textvariable=self.progress_interval_var,
                width=80,
                height=30,
                state="disabled"
            )
            self.progress_interval_entry.pack(side="left")

            seconds_label = ctk.CTkLabel(interval_frame, text="seconds")
            seconds_label.pack(side="left", padx=(5, 0))

            self.progress_desc = ctk.CTkLabel(
                progress_frame,
                text="Disabled - Enable to receive periodic progress updates with statistics and graphs",
                text_color="gray",
                wraplength=400,
                font=ctk.CTkFont(size=11)
            )
            self.progress_desc.grid(row=2, column=0, columnspan=3, pady=(5, 0), sticky="w")
            
            save_btn = ctk.CTkButton(
                discord_card,
                text="Save Discord Settings",
                command=self.save_discord_settings,
                height=35,
                font=ctk.CTkFont(size=14, weight="bold")
            )
            save_btn.grid(row=7, column=0, padx=20, pady=20, sticky="w")


    def toggle_progress_options(self):
        if self.progress_enabled_var.get():
            self.progress_interval_entry.configure(state="normal")
            interval_minutes = int(self.progress_interval_var.get()) / 60
            self.progress_desc.configure(
                text=f"Enabled - Will send progress graphs every {interval_minutes:.1f} minutes with statistics and performance charts",
                text_color="white"
            )
        else:
            self.progress_interval_entry.configure(state="disabled")
            self.progress_desc.configure(
                text="Disabled - Enable to receive periodic progress updates with statistics and graphs",
                text_color="gray"
            )


    def save_discord_settings(self):
        webhook_url = self.webhook_entry.get().strip()
        role_id = self.UserId_entry.get().strip()
        progress_enabled = self.progress_enabled_var.get()

        try:
            progress_interval = int(self.progress_interval_var.get())
            if progress_interval < 60:
                messagebox.showwarning("Invalid Interval", "Progress interval must be at least 60 seconds")
                return
        except ValueError:
            messagebox.showwarning("Invalid Interval", "Please enter a valid number for the progress interval")
            return
        
        self.launcher.set_discord_webhook(webhook_url, role_id)
        self.launcher.set_progress_tracking(progress_enabled, progress_interval)

        interval_minutes = int(self.progress_interval_var.get()) / 60
        self.progress_desc.configure(
                text=f"Enabled - Will send progress graphs every {interval_minutes:.1f} minutes with statistics and performance charts",
                text_color="white"
            )
        self.config["webhook_url"] = webhook_url
        self.config["user_id"] = role_id
        self.config["progress_enabled"] = progress_enabled
        self.config["progress_interval"] = progress_interval
        save_config(self.config)

        if webhook_url:
            if progress_enabled:
                messagebox.showinfo("Saved", f"Discord settings saved! Progress tracking enabled - updates every {progress_interval/60:.1f} minutes")
            else:
                messagebox.showinfo("Saved", "Discord settings saved successfully!")
            self.log_message("Discord settings saved")
        else:
            messagebox.showinfo("Saved", "Discord notifications disabled (no webhook URL).")
            self.log_message("Discord notifications disabled")

    def test_webhook(self):
        webhook_url = self.webhook_entry.get().strip()
        role_id = self.UserId_entry.get().strip()

        if not webhook_url:
            messagebox.showwarning("Missing Webhook", "Please enter a Discord webhook URL first.")
            return
        

        self.launcher.set_discord_webhook(webhook_url, role_id)

        success = self.launcher.discord_webhook.send_notification(
            "STP>=1",
            "STP = 1",
            "https://www.roblox.com/games/6161235818"
        )

        if success:
            messagebox.showinfo("Success", "Test webhook sent successfully!")
            self.log_message("Test webhook sent successfully")
        else:
            messagebox.showerror("Error", "Failed to send test webhook. Check the URL and try again.")
            self.log_message("Failed to send test webhook")


    def create_parameter_card(self):
        param_card = ctk.CTkFrame(self.content_frame)
        param_card.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(0, 20))
        param_card.grid_columnconfigure(1, weight=1)
        
        title = ctk.CTkLabel(
            param_card,
            text="Parameter Conditions",
            font=ctk.CTkFont(size=18, weight="bold")
        )
        title.grid(row=0, column=0, padx=20, pady=(20, 10), sticky="w")

        desc = ctk.CTkLabel(
            param_card,
            text="Set conditions that servers must meet. The script will stop when ALL conditions are satisfied.",
            wraplength=400
        )
        desc.grid(row=1, column=0, columnspan=2, padx=20, pady=5, sticky="w")

        self.conditions_display = ctk.CTkLabel(
            param_card,
            text="No conditions set",
            text_color="gray",
            wraplength=400
        )
        self.conditions_display.grid(row=2, column=0, columnspan=2, padx=20, pady=10, sticky="w")
        
        param_btn = ctk.CTkButton(
            param_card,
            text="Configure Parameters",
            command=self.open_parameter_selector,
            height=35,
            font=ctk.CTkFont(size=14, weight="bold")
        )
        param_btn.grid(row=3, column=0, padx=20, pady=20, sticky="w")

    def create_statistics_cards(self):
        stats_frame = ctk.CTkFrame(self.content_frame, fg_color="transparent")
        stats_frame.grid(row=3, column=0, columnspan=2, sticky="ew", pady=20)
        stats_frame.grid_columnconfigure((0, 1, 2, 3), weight=1)
        
        servers_card = ctk.CTkFrame(stats_frame)
        servers_card.grid(row=0, column=0, padx=10, pady=10, sticky="ew")
        
        ctk.CTkLabel(
            servers_card,
            text="Servers Joined",
            font=ctk.CTkFont(size=12)
        ).pack(pady=(20, 5))
        
        self.servers_stat = ctk.CTkLabel(
            servers_card,
            text="0",
            font=ctk.CTkFont(size=24, weight="bold")
        )
        self.servers_stat.pack(pady=(0, 20))

        data_card = ctk.CTkFrame(stats_frame)
        data_card.grid(row=0, column=1, padx=10, pady=10, sticky="ew")
        
        ctk.CTkLabel(
            data_card,
            text="Data Collected",
            font=ctk.CTkFont(size=12)
        ).pack(pady=(20, 5))
        
        self.data_stat = ctk.CTkLabel(
            data_card,
            text="0",
            font=ctk.CTkFont(size=24, weight="bold")
        )
        self.data_stat.pack(pady=(0, 20))

        rate_card = ctk.CTkFrame(stats_frame)
        rate_card.grid(row=0, column=2, padx=10, pady=10, sticky="ew")
        
        ctk.CTkLabel(
            rate_card,
            text="Success Rate",
            font=ctk.CTkFont(size=12)
        ).pack(pady=(20, 5))
        
        self.rate_stat = ctk.CTkLabel(
            rate_card,
            text="0%",
            font=ctk.CTkFont(size=24, weight="bold")
        )
        self.rate_stat.pack(pady=(0, 20))

        avg_card = ctk.CTkFrame(stats_frame)
        avg_card.grid(row=0, column=3, padx=10, pady=10, sticky="ew")
        
        ctk.CTkLabel(
            avg_card,
            text="Avg Server Time",
            font=ctk.CTkFont(size=12)
        ).pack(pady=(20, 5))
        
        self.avg_time_stat = ctk.CTkLabel(
            avg_card,
            text="0.0s",
            font=ctk.CTkFont(size=24, weight="bold")
        )
        self.avg_time_stat.pack(pady=(0, 20))

    def create_logs_section(self):
        logs_card = ctk.CTkFrame(self.content_frame)
        logs_card.grid(row=4, column=0, columnspan=2, sticky="ew", pady=20)
        logs_card.grid_rowconfigure(1, weight=1)
        logs_card.grid_columnconfigure(0, weight=1)
        
        title = ctk.CTkLabel(
            logs_card,
            text="Activity Logs",
            font=ctk.CTkFont(size=18, weight="bold")
        )
        title.grid(row=0, column=0, padx=20, pady=(20, 10), sticky="w")
        
        self.log_text = ctk.CTkTextbox(
            logs_card,
            height=200,
            font=ctk.CTkFont(family="Consolas", size=11)
        )
        self.log_text.grid(row=1, column=0, padx=20, pady=(0, 20), sticky="ew")
        
        self.log_message("System initialized")
        self.log_message("Ready to start launcher")

    def open_parameter_selector(self):


        data_format = deepcopy(Macro.Data.FORMAT)

        dialog = ParameterSelectorDialog(self.root, data_format)
        self.root.wait_window(dialog.dialog)
        
        conditions = dialog.get_conditions()
        self.launcher.set_parameters(conditions)
        
        if conditions:
            condition_strings = []
            for condition in conditions:
                condition_strings.append(f"{condition['parameter']} {condition['operator']} {condition['value']}")
            display_text = "; ".join(condition_strings)
        else:
            display_text = "No conditions set"
        self.config["parameters"] = conditions
        save_config(self.config)
        self.conditions_display.configure(text=display_text)
        self.log_message(f"Parameter conditions updated: {display_text}")

    def on_match_found(self):
        self.start_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")
        self.log_message("Match found! Launcher stopped automatically.")

    def start_launcher(self):
        code = self.url_entry.get().strip()
        if not code:
            messagebox.showerror("Error", "Please enter a private server code")
            return
        
        if not validate_private_server_code(code):
            messagebox.showerror(
                "Invalid Code", 
                "Please enter a valid 32-character private server code.\n\n"
                "Example: 12345678912345678912345678912345"
            )
            return
        
        full_url = build_private_server_url(code)
        
        if not self.launcher.validator.conditions:
            response = messagebox.askyesno(
                "No Conditions Set", 
                "No parameter conditions are set. The script will run indefinitely until manually stopped. Continue?"
            )
            if not response:
                return
            
        self.start_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        self.log_message("Starting launcher...")
        self.config["private_server_url"] = code  
        save_config(self.config)
        self.root.iconify()
        threading.Thread(
            target=self.launcher.main_loop,
            args=(full_url,), 
            daemon=True
        ).start()


    def stop_launcher(self):
        self.launcher.is_running = False
        with self.launcher._queue_lock:
            self.launcher.queue.clear()
        
        self.launcher.pause_event.clear()
        self.start_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")
        self.log_message("Stopping launcher...")

    def update_statistics(self, stats):
        if self.servers_stat:
            self.servers_stat.configure(text=str(stats["servers"]))
        if self.data_stat:
            self.data_stat.configure(text=str(stats["data"]))
        if self.rate_stat:
            self.rate_stat.configure(text=stats["rate"])
        if hasattr(self, "avg_time_stat") and "avg_time" in stats:
            self.avg_time_stat.configure(text=stats["avg_time"])



    def log_message(self, message):
            timestamp = time.strftime("%H:%M:%S")
            
            stack = inspect.stack()
            
            caller_info = ""
            try:
                if len(stack) > 3:
                    frame = stack[3]
                    lineno = frame.lineno
                    function = frame.function
                    caller_info = f" [{function}:{lineno}]"
                elif len(stack) > 1:
                    frame = stack[min(len(stack)-1, 3)]
                    lineno = frame.lineno
                    function = frame.function
                    caller_info = f" [{function}:{lineno}]"
            except Exception:
                pass 
            
            formatted_message = f"[{timestamp}]{caller_info} {message}\n"
            self.root.after(0, lambda: self._update_log_text(formatted_message))
        
    def _update_log_text(self, message):
        self.log_text.insert("end", message)
        self.log_text.see("end")

    def run(self):
        self.log_message("Application started")
        self.root.mainloop()


if __name__ == "__main__":
    try:
        app = UI()
        app.run()
    except Exception as e:
        pass