"""
Main Application Entry Point.
Coordinates Asynchronous Camera Input, MediaPipe/YOLO Hand Detection,
Adaptive One Euro Filtering, Gesture Recognition, 60 FPS Game Mechanics,
Screen Navigation, Live Telemetry Diagnostics, and Persistence.
"""

import os
import sys
import time
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np
import pygame

from src.camera.camera_manager import CameraManager, CameraStatus
from src.data.statistics_store import StatisticsStore
from src.detection.hand_detector import BaseHandDetector, DetectedHand
from src.detection.hand_tracker import HandTracker, TrackingStatus
from src.detection.mediapipe_detector import MediaPipeHandDetector
from src.detection.vision_worker import VisionWorker
from src.detection.yolo_detector import YOLOHandDetector
from src.game.game_controller import GameController
from src.game.game_engine import GameEngine, GameState
from src.gestures.gesture_recognizer import GestureRecognizer, RecognizedGesture
from src.ui.calibration_screen import CalibrationScreen
from src.ui.game_screen import GameScreenUI
from src.ui.how_to_play_screen import HowToPlayScreen
from src.ui.menu import MainMenuScreen
from src.ui.practice_screen import PracticeScreen
from src.ui.results_screen import ResultsScreen
from src.ui.settings_screen import SettingsScreen
from src.ui.stats_screen import StatisticsScreen
from src.utils.config import ConfigManager
from src.utils.logging_config import setup_logger

logger = setup_logger("MainApp")


class AppScreen(Enum):
    MAIN_MENU = "Main Menu"
    PLAYING = "Playing"
    PRACTICE = "Practice"
    CALIBRATION = "Calibration"
    HOW_TO_PLAY = "How To Play"
    SETTINGS = "Settings"
    STATISTICS = "Statistics"
    RESULTS = "Results"


class CyberSliceApp:
    """Master application coordinator managing asynchronous vision pipeline and active screens."""

    def __init__(self):
        # 1. Initialize Pygame
        pygame.init()
        pygame.font.init()

        # 2. Load Configuration & Persistence
        self.config = ConfigManager("config/settings.json")
        self.stats = StatisticsStore(self.config.get("storage", "stats_file", "data/game_stats.json"))

        # Screen dimensions
        self.width = self.config.get("game", "window_width", 1280)
        self.height = self.config.get("game", "window_height", 720)
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("CYBER SLICE: CV Hand Gesture Game")
        self.clock = pygame.time.Clock()
        self.target_fps = self.config.get("game", "target_fps", 60)

        # 3. Typography
        self.fonts = self._load_fonts()

        # 4. Initialize Camera Manager (Module 1)
        cam_cfg = self.config.get("camera")
        self.camera = CameraManager(
            device_index=cam_cfg.get("device_index", 0),
            width=cam_cfg.get("width", 640),
            height=cam_cfg.get("height", 480),
            fps=cam_cfg.get("fps", 30),
            mirrored=cam_cfg.get("mirrored", True),
            auto_reconnect=cam_cfg.get("auto_reconnect", True)
        )
        self.camera.start()

        # 5. Initialize Hand Detection Module (Module 2)
        self.detector = self._create_detector()

        # 6. Hand Target Tracker with Adaptive One Euro Filter
        gest_cfg = self.config.get("gestures") or {}
        cam_cfg = self.config.get("camera") or {}
        self.hand_tracker = HandTracker(
            preferred_handedness="Right",
            control_point=gest_cfg.get("control_point", "index_tip"),
            tracking_loss_timeout=gest_cfg.get("tracking_loss_timeout", 0.14),
            min_cutoff=gest_cfg.get("one_euro_min_cutoff", 1.2),
            beta=gest_cfg.get("one_euro_beta", 15.0),
            sensitivity=gest_cfg.get("sensitivity", 1.0),
            is_mirrored=cam_cfg.get("mirrored", True)
        )

        # 7. Initialize Gesture Recognition Module (Module 3)
        self.gesture_recognizer = GestureRecognizer(
            sensitivity=gest_cfg.get("sensitivity", 1.0),
            cooldown_seconds=gest_cfg.get("cooldown_seconds", 0.35),
            swipe_min_distance=gest_cfg.get("swipe_min_distance", 0.16),
            pinch_threshold=gest_cfg.get("pinch_threshold", 0.055),
            action_mapping=self.config.get("game", "action_mapping")
        )

        # 8. Start Asynchronous Vision Pipeline Worker
        self.vision_worker = VisionWorker(
            camera_manager=self.camera,
            detector=self.detector,
            hand_tracker=self.hand_tracker,
            gesture_recognizer=self.gesture_recognizer
        )
        self.vision_worker.start()

        # 9. Initialize Game Engine & Controller (Module 4)
        game_cfg = self.config.get("game", {})
        self.slicing_mode = game_cfg.get("slicing_mode", "motion")
        self.game = GameEngine(
            width=self.width,
            height=self.height,
            initial_lives=game_cfg.get("initial_lives", 3),
            difficulty=game_cfg.get("difficulty", "Medium"),
            sound_enabled=game_cfg.get("sound_enabled", True),
            blade_trail_length=game_cfg.get("blade_trail_length", 16)
        )
        # Seed engine with lifetime high score from persistent store
        summary = self.stats.get_summary()
        self.game.high_score = summary.get("highest_score", 0)

        self.controller = GameController(
            game_engine=self.game,
            action_mapping=game_cfg.get("action_mapping")
        )

        # 10. Initialize UI Screens (Module 5)
        self.screen_state = AppScreen.MAIN_MENU
        self._init_screens()

        # Runtime Tracking
        self.is_running = True
        self.last_results_data = {}
        self.current_gesture = RecognizedGesture(
            name="Unknown", confidence=0.0, description="Scanning...",
            action="none", cursor_pos=(0.5, 0.5)
        )

    def _load_fonts(self) -> Dict[str, pygame.font.Font]:
        """Loads modern clean typography with fallback to standard system sans-serif."""
        font_names = ["Segoe UI", "Trebuchet MS", "Arial", "Helvetica", "Roboto", None]
        def get_f(size: int, bold: bool = False):
            for name in font_names:
                try:
                    f = pygame.font.SysFont(name, size, bold=bold)
                    if f:
                        return f
                except Exception:
                    continue
            return pygame.font.Font(None, size)

        return {
            "title": get_f(48, bold=True),
            "large": get_f(34, bold=True),
            "medium": get_f(22, bold=True),
            "small": get_f(15, bold=False)
        }

    def _create_detector(self) -> BaseHandDetector:
        """Instantiates either YOLO or MediaPipe based on configuration."""
        backend = self.config.get("detection", "backend", "mediapipe").lower()
        if backend == "yolo":
            model_path = self.config.get("detection", "yolo_model_path", "models/yolov8n-hand.pt")
            conf = self.config.get("detection", "yolo_confidence_threshold", 0.5)
            logger.info("Initializing YOLO detector (%s)...", model_path)
            return YOLOHandDetector(model_path=model_path, confidence_threshold=conf)
        else:
            logger.info("Initializing MediaPipe Hands detector (ultra-fast model_complexity=0)...")
            return MediaPipeHandDetector(
                max_num_hands=self.config.get("detection", "max_num_hands", 2),
                min_detection_confidence=self.config.get("detection", "min_detection_confidence", 0.65),
                min_tracking_confidence=self.config.get("detection", "min_tracking_confidence", 0.55),
                model_complexity=0,
                is_mirrored=self.config.get("camera", "mirrored", True)
            )

    def _init_screens(self) -> None:
        """Instantiates all interface view components."""
        self.menu_screen = MainMenuScreen(
            screen_w=self.width,
            screen_h=self.height,
            fonts=self.fonts,
            on_start_game=self.start_game,
            on_practice=lambda: self.switch_screen(AppScreen.PRACTICE),
            on_calibrate=lambda: self.switch_screen(AppScreen.CALIBRATION),
            on_how_to_play=lambda: self.switch_screen(AppScreen.HOW_TO_PLAY),
            on_settings=lambda: self.switch_screen(AppScreen.SETTINGS),
            on_stats=lambda: self.switch_screen(AppScreen.STATISTICS),
            on_exit=self.exit_app
        )

        self.game_ui = GameScreenUI(
            screen_w=self.width,
            screen_h=self.height,
            fonts=self.fonts,
            on_resume=self.game.toggle_pause,
            on_restart=self.game.reset_round,
            on_menu=lambda: self.switch_screen(AppScreen.MAIN_MENU)
        )

        self.practice_screen = PracticeScreen(
            screen_w=self.width,
            screen_h=self.height,
            fonts=self.fonts,
            on_back=lambda: self.switch_screen(AppScreen.MAIN_MENU)
        )

        self.calibration_screen = CalibrationScreen(
            screen_w=self.width,
            screen_h=self.height,
            fonts=self.fonts,
            config_manager=self.config,
            hand_tracker=self.hand_tracker,
            on_back=self._on_calibration_saved
        )

        self.how_to_play_screen = HowToPlayScreen(
            screen_w=self.width,
            screen_h=self.height,
            fonts=self.fonts,
            on_back=lambda: self.switch_screen(AppScreen.MAIN_MENU)
        )

        self.settings_screen = SettingsScreen(
            screen_w=self.width,
            screen_h=self.height,
            fonts=self.fonts,
            config_manager=self.config,
            on_back=self._on_settings_saved
        )

        self.stats_screen = StatisticsScreen(
            screen_w=self.width,
            screen_h=self.height,
            fonts=self.fonts,
            stats_store=self.stats,
            on_back=lambda: self.switch_screen(AppScreen.MAIN_MENU)
        )

        self.results_screen = ResultsScreen(
            screen_w=self.width,
            screen_h=self.height,
            fonts=self.fonts,
            on_play_again=self.start_game,
            on_menu=lambda: self.switch_screen(AppScreen.MAIN_MENU),
            on_stats=lambda: self.switch_screen(AppScreen.STATISTICS)
        )

    @property
    def diagnostics(self):
        """Exposes the live diagnostics HUD panel for telemetry inspection."""
        return self.game_ui.diagnostics

    def _on_calibration_saved(self) -> None:
        """Applies calibration settings changes."""
        self.slicing_mode = self.config.get("game", "slicing_mode", "motion")
        sens = self.config.get("gestures", "sensitivity", 1.0)
        self.gesture_recognizer.set_sensitivity(sens)
        self.hand_tracker.update_settings(
            control_point=self.config.get("gestures", "control_point", "index_tip"),
            min_cutoff=self.config.get("gestures", "one_euro_min_cutoff", 1.2),
            beta=self.config.get("gestures", "one_euro_beta", 15.0),
            timeout=self.config.get("gestures", "tracking_loss_timeout", 0.14),
            sensitivity=sens,
            is_mirrored=self.config.get("camera", "mirrored", True)
        )
        self.switch_screen(AppScreen.MAIN_MENU)

    def _on_settings_saved(self) -> None:
        """Applies configuration changes to active camera, detector, and gameplay."""
        cam_idx = self.config.get("camera", "device_index", 0)
        self.camera.set_device_index(cam_idx)
        is_mirrored = self.config.get("camera", "mirrored", True)
        self.camera.set_mirrored(is_mirrored)
        if hasattr(self.detector, "set_mirrored"):
            self.detector.set_mirrored(is_mirrored)

        backend = self.config.get("detection", "backend", "mediapipe").lower()
        if (backend == "yolo" and not isinstance(self.detector, YOLOHandDetector)) or \
           (backend == "mediapipe" and not isinstance(self.detector, MediaPipeHandDetector)):
            self.vision_worker.stop()
            self.detector.release()
            self.detector = self._create_detector()
            self.vision_worker.detector = self.detector
            self.vision_worker.start()

        sens = self.config.get("gestures", "sensitivity", 1.0)
        self.gesture_recognizer.set_sensitivity(sens)
        self.hand_tracker.update_settings(
            sensitivity=sens,
            is_mirrored=is_mirrored
        )

        diff = self.config.get("game", "difficulty", "Medium")
        self.game._setup_difficulty(diff)
        self.game.audio.enabled = self.config.get("game", "sound_enabled", True)

        self.switch_screen(AppScreen.MAIN_MENU)

    def switch_screen(self, screen: AppScreen) -> None:
        """Transitions to the specified application screen."""
        self.screen_state = screen
        logger.info("Navigated to screen: %s", screen.value)

    def start_game(self) -> None:
        """Launches a fresh match round and switches to PLAYING screen."""
        self.game.reset_round()
        if hasattr(self, "controller") and self.controller:
            self.controller.last_triggered_action = "none"
        if hasattr(self, "gesture_recognizer") and self.gesture_recognizer:
            self.gesture_recognizer.reset()
        self.last_results_data = {}
        self.switch_screen(AppScreen.PLAYING)

    def _on_game_over(self, results: Optional[Dict[str, Any]] = None) -> None:
        """Handles match completion, persists stats, and transitions to RESULTS screen."""
        if results is not None:
            self.last_results_data = results
        else:
            res = self.stats.record_game(
                score=self.game.score,
                fruits_sliced=self.game.fruits_sliced,
                bombs_hit=self.game.bombs_hit,
                duration_seconds=self.game.game_duration,
                difficulty=self.game.difficulty
            )
            self.last_results_data = {
                "score": self.game.score,
                "fruits_sliced": self.game.fruits_sliced,
                "bombs_hit": self.game.bombs_hit,
                "missed_fruits": self.game.missed_fruits,
                "duration": self.game.game_duration,
                "difficulty": self.game.difficulty,
                "is_high_score": res.get("is_new_high", False),
                "lifetime_high": res.get("highest_score", self.game.score)
            }
        self.switch_screen(AppScreen.RESULTS)

    def exit_app(self) -> None:
        """Stops loops and initiates clean shutdown."""
        self.is_running = False

    def run(self) -> None:
        """Primary application frame execution loop running at full 60 FPS."""
        logger.info("Cyber Slice rendering loop started at %d FPS.", self.target_fps)

        while self.is_running:
            dt = self.clock.tick(self.target_fps) / 1000.0
            dt = min(0.066, dt)

            # 1. Fetch Fresh Vision State from Background Worker (<0.05ms)
            v_state = self.vision_worker.get_latest_state()
            raw_frame = v_state.raw_frame
            detected_hands = v_state.detected_hands
            self.current_gesture = v_state.gesture

            # Measure real-time input-to-render latency
            now_t = time.time()
            if v_state.capture_timestamp > 0:
                measured_latency_ms = (now_t - v_state.capture_timestamp) * 1000.0
            else:
                measured_latency_ms = 30.0

            # 2. Dispatch gestures to active controller
            is_lost = (v_state.tracking_status == TrackingStatus.LOST)

            if self.screen_state == AppScreen.PLAYING:
                self.controller.process_gesture(
                    self.current_gesture,
                    is_tracking_lost=is_lost,
                    slicing_mode=self.slicing_mode
                )
            elif self.screen_state == AppScreen.PRACTICE:
                if not is_lost:
                    self.practice_screen.record_gesture(self.current_gesture.name)
                    self.stats.record_practice_gesture(self.current_gesture.name)

            # Gesture Shortcuts for menus
            if self.current_gesture.is_trigger:
                if self.screen_state == AppScreen.MAIN_MENU and self.current_gesture.name == "Open Palm":
                    self.start_game()

            # 3. Handle Pygame OS Events
            mouse_pos = pygame.mouse.get_pos()

            # Virtual Hand Cursor for non-game menus
            if not is_lost and self.screen_state != AppScreen.PLAYING:
                hand_cursor_px = (
                    int(self.current_gesture.cursor_pos[0] * self.width),
                    int(self.current_gesture.cursor_pos[1] * self.height)
                )
                interact_pos = hand_cursor_px
            else:
                interact_pos = mouse_pos

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.exit_app()

                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        if self.screen_state == AppScreen.PLAYING:
                            self.game.toggle_pause()
                        else:
                            self.switch_screen(AppScreen.MAIN_MENU)
                    elif event.key == pygame.K_p and self.screen_state == AppScreen.PLAYING:
                        self.game.toggle_pause()
                    elif event.key == pygame.K_r and self.screen_state == AppScreen.PLAYING:
                        self.game.reset_round()
                    elif event.key == pygame.K_d:
                        self.game_ui.diagnostics.toggle()
                    elif event.key == pygame.K_SPACE:
                        if self.screen_state == AppScreen.MAIN_MENU:
                            self.start_game()

                # Dispatch events to active view
                if self.screen_state == AppScreen.MAIN_MENU:
                    self.menu_screen.handle_event(event)
                elif self.screen_state == AppScreen.PLAYING:
                    self.game_ui.handle_event(event, is_paused=(self.game.state == GameState.PAUSED))
                elif self.screen_state == AppScreen.PRACTICE:
                    self.practice_screen.handle_event(event)
                elif self.screen_state == AppScreen.CALIBRATION:
                    self.calibration_screen.handle_event(event)
                elif self.screen_state == AppScreen.HOW_TO_PLAY:
                    self.how_to_play_screen.handle_event(event)
                elif self.screen_state == AppScreen.SETTINGS:
                    self.settings_screen.handle_event(event)
                elif self.screen_state == AppScreen.STATISTICS:
                    self.stats_screen.handle_event(event)
                elif self.screen_state == AppScreen.RESULTS:
                    self.results_screen.handle_event(event)

            # Pinch gesture clicks hovered buttons on menus
            if self.current_gesture.name == "Pinch Gesture" and self.current_gesture.is_trigger:
                if self.screen_state == AppScreen.MAIN_MENU:
                    self.menu_screen.trigger_hovered_or_first(interact_pos)
                elif self.screen_state == AppScreen.PLAYING and self.game.state == GameState.PAUSED:
                    self.game_ui.trigger_hovered_button(interact_pos)
                elif self.screen_state == AppScreen.RESULTS:
                    self.results_screen.trigger_hovered_button(interact_pos)
                elif self.screen_state == AppScreen.SETTINGS:
                    self.settings_screen.trigger_hovered_button(interact_pos)
                elif self.screen_state == AppScreen.STATISTICS:
                    self.stats_screen.trigger_hovered_button(interact_pos)
                elif self.screen_state == AppScreen.HOW_TO_PLAY:
                    self.how_to_play_screen.trigger_hovered_button(interact_pos)
                elif self.screen_state == AppScreen.PRACTICE:
                    self.practice_screen.trigger_hovered_button(interact_pos)
                elif self.screen_state == AppScreen.CALIBRATION:
                    self.calibration_screen.trigger_hovered_button(interact_pos)

            # 4. Update Active View
            if self.screen_state == AppScreen.MAIN_MENU:
                self.menu_screen.update(interact_pos, dt)

            elif self.screen_state == AppScreen.PLAYING:
                self.game.update(dt)
                self.game_ui.update(interact_pos, dt, is_paused=(self.game.state == GameState.PAUSED))

                # Check for Game Over transition
                if self.game.state == GameState.GAME_OVER:
                    self._on_game_over()

            elif self.screen_state == AppScreen.PRACTICE:
                self.practice_screen.update(interact_pos, dt)

            elif self.screen_state == AppScreen.CALIBRATION:
                self.calibration_screen.update(interact_pos, dt)

            elif self.screen_state == AppScreen.HOW_TO_PLAY:
                self.how_to_play_screen.update(interact_pos, dt)

            elif self.screen_state == AppScreen.SETTINGS:
                self.settings_screen.update(interact_pos, dt)

            elif self.screen_state == AppScreen.STATISTICS:
                self.stats_screen.update(interact_pos, dt)

            elif self.screen_state == AppScreen.RESULTS:
                self.results_screen.update(interact_pos, dt)

            # 5. Render Active View
            if self.screen_state == AppScreen.MAIN_MENU:
                self.menu_screen.draw(
                    surface=self.screen,
                    camera_status=self.camera.status.value,
                    fps=v_state.camera_fps,
                    backend_name="MediaPipe Hands" if isinstance(self.detector, MediaPipeHandDetector) else "YOLOv8",
                    recognized_gesture=self.current_gesture.name
                )

            elif self.screen_state == AppScreen.PLAYING:
                self.screen.fill((14, 18, 30))
                # Render fruits, blade trail, particles
                self.game.render(self.screen, self.fonts["title"], self.fonts["medium"], self.fonts["small"])
                # Render HUD, PIP, and Live Telemetry Diagnostics
                self.game_ui.draw_hud(
                    surface=self.screen,
                    game=self.game,
                    raw_frame=raw_frame,
                    detected_hands=detected_hands,
                    gesture=self.current_gesture,
                    camera_fps=v_state.camera_fps,
                    backend_name="MediaPipe" if isinstance(self.detector, MediaPipeHandDetector) else "YOLO",
                    processing_fps=v_state.processing_fps,
                    inference_time_ms=v_state.inference_time_ms,
                    gesture_time_ms=v_state.gesture_time_ms,
                    latency_ms=measured_latency_ms,
                    tracking_status=v_state.tracking_status,
                    hand_confidence=v_state.hand_confidence
                )

            elif self.screen_state == AppScreen.PRACTICE:
                self.practice_screen.draw(
                    surface=self.screen,
                    raw_frame=raw_frame,
                    detected_hands=detected_hands,
                    current_gesture=self.current_gesture,
                    camera_fps=v_state.camera_fps
                )

            elif self.screen_state == AppScreen.CALIBRATION:
                self.calibration_screen.draw(
                    surface=self.screen,
                    raw_frame=raw_frame,
                    detected_hands=detected_hands,
                    gesture=self.current_gesture,
                    tracking_status=v_state.tracking_status,
                    camera_fps=v_state.camera_fps,
                    latency_ms=measured_latency_ms
                )

            elif self.screen_state == AppScreen.HOW_TO_PLAY:
                self.how_to_play_screen.draw(self.screen)

            elif self.screen_state == AppScreen.SETTINGS:
                self.settings_screen.draw(self.screen)

            elif self.screen_state == AppScreen.STATISTICS:
                self.stats_screen.draw(self.screen)

            elif self.screen_state == AppScreen.RESULTS:
                d = self.last_results_data
                self.results_screen.draw(
                    surface=self.screen,
                    score=d.get("score", 0),
                    fruits_sliced=d.get("fruits_sliced", 0),
                    bombs_hit=d.get("bombs_hit", 0),
                    missed_fruits=d.get("missed_fruits", 0),
                    duration_seconds=d.get("duration", 0.0),
                    difficulty=d.get("difficulty", "Medium"),
                    is_new_high_score=d.get("is_high_score", False),
                    lifetime_high=d.get("lifetime_high", 0)
                )

            # Virtual cursor for non-game menus
            if not is_lost and self.screen_state != AppScreen.PLAYING:
                cx_px = int(self.current_gesture.cursor_pos[0] * self.width)
                cy_px = int(self.current_gesture.cursor_pos[1] * self.height)
                pygame.draw.circle(self.screen, (0, 235, 255), (cx_px, cy_px), 8, 2)
                pygame.draw.circle(self.screen, (255, 255, 255), (cx_px, cy_px), 4)

            # Flip to screen
            pygame.display.flip()

        # Shutdown & Cleanup
        self._shutdown()

    def _shutdown(self) -> None:
        """Releases all worker, camera, vision, and window resources safely."""
        logger.info("Initiating safe application shutdown...")
        if hasattr(self, "vision_worker") and self.vision_worker:
            self.vision_worker.stop()
        if hasattr(self, "camera") and self.camera:
            self.camera.stop()
        if hasattr(self, "detector") and self.detector:
            self.detector.release()
        self.config.save()
        pygame.quit()
        logger.info("Application cleanly terminated.")


def main():
    """Application entry point."""
    try:
        app = CyberSliceApp()
        app.run()
    except Exception as e:
        logger.critical("Fatal application error: %s", e, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
