"""
Integration test for full application lifecycle.
Verifies that CyberSliceApp starts cleanly, initializes all modules, transitions screens, and shuts down without errors.
"""

import os
import pygame
import pytest

from main import CyberSliceApp, AppScreen


class TestAppLifecycle:

    @pytest.fixture(autouse=True)
    def setup_pygame_dummy(self):
        """Run pygame headless with dummy video driver for headless testing."""
        os.environ["SDL_VIDEODRIVER"] = "dummy"
        pygame.init()
        yield
        pygame.quit()

    def test_app_initialization_and_state(self):
        """Verifies CyberSliceApp initializes all subsystems without exceptions."""
        app = CyberSliceApp()
        assert app.screen_state == AppScreen.MAIN_MENU
        assert app.vision_worker is not None
        assert app.hand_tracker is not None
        assert app.gesture_recognizer is not None
        assert app.calibration_screen is not None
        assert app.diagnostics is not None
        app._shutdown()

    def test_screen_transitions(self):
        """Verifies clean transitions between Main Menu, Calibration, Game, and Results."""
        app = CyberSliceApp()
        assert app.screen_state == AppScreen.MAIN_MENU

        # Navigate to Calibration
        app.switch_screen(AppScreen.CALIBRATION)
        assert app.screen_state == AppScreen.CALIBRATION

        # Return to Menu
        app.switch_screen(AppScreen.MAIN_MENU)
        assert app.screen_state == AppScreen.MAIN_MENU

        # Start Game
        app.start_game()
        assert app.screen_state == AppScreen.PLAYING

        # End Game -> Results
        app._on_game_over({"score": 150, "fruits_sliced": 25, "accuracy": 92.5})
        assert app.screen_state == AppScreen.RESULTS

        # Results -> Back to Menu
        app.switch_screen(AppScreen.MAIN_MENU)
        assert app.screen_state == AppScreen.MAIN_MENU

        app._shutdown()

    def test_diagnostics_toggle(self):
        """Verifies 'D' key toggles live diagnostics HUD."""
        app = CyberSliceApp()
        assert app.diagnostics.enabled is True
        app.diagnostics.toggle()
        assert app.diagnostics.enabled is False
        app.diagnostics.toggle()
        assert app.diagnostics.enabled is True
        app._shutdown()

    def test_calibration_screen_render(self):
        """Verifies calibration screen renders successfully with active detected hand landmarks."""
        import numpy as np
        from src.detection.hand_detector import DetectedHand, HandLandmark
        from src.gestures.gesture_recognizer import RecognizedGesture

        app = CyberSliceApp()
        dummy_surface = pygame.Surface((1280, 720))
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        dummy_hand = DetectedHand(
            landmarks=[HandLandmark(x=0.5, y=0.5, z=0.0) for _ in range(21)],
            bbox=(200, 150, 200, 200),
            handedness="Right",
            confidence=0.95,
            hand_id=0
        )
        dummy_gesture = RecognizedGesture(
            name="Open Palm",
            confidence=0.92,
            description="All 5 fingers extended",
            action="start_game",
            cursor_pos=(0.5, 0.5),
            handedness="Right"
        )
        from src.detection.hand_tracker import TrackingStatus
        app.calibration_screen.draw(
            surface=dummy_surface,
            raw_frame=dummy_frame,
            detected_hands=[dummy_hand],
            gesture=dummy_gesture,
            tracking_status=TrackingStatus.TRACKING,
            camera_fps=30.0,
            latency_ms=25.0
        )
        app._shutdown()

    def test_repeated_play_again_resets(self):
        """
        Verifies that 'Play Again' works repeatedly for at least 5 consecutive rounds.
        Ensures score, timer, fruits, bombs, lives, particles, and trails reset cleanly
        without leaks or duplicate loops, while persistent high scores remain intact.
        """
        app = CyberSliceApp()
        initial_high_score = app.game.high_score

        for round_idx in range(1, 6):
            # 1. Start or Play Again
            if round_idx == 1:
                app.start_game()
            else:
                # Trigger Play Again from results screen
                assert app.screen_state == AppScreen.RESULTS
                app.results_screen.btn_play_again.trigger()

            # Verify clean reset state on round start
            assert app.screen_state == AppScreen.PLAYING
            assert app.game.score == 0, f"Round {round_idx}: Score did not reset"
            assert app.game.fruits_sliced == 0, f"Round {round_idx}: Fruits sliced did not reset"
            assert app.game.bombs_hit == 0, f"Round {round_idx}: Bombs hit did not reset"
            assert app.game.missed_fruits == 0, f"Round {round_idx}: Missed fruits did not reset"
            assert app.game.lives == app.game.initial_lives, f"Round {round_idx}: Lives did not reset"
            assert app.game.game_duration == 0.0, f"Round {round_idx}: Game duration did not reset"
            assert len(app.game.fruits) == 0, f"Round {round_idx}: Lingering fruits"
            assert len(app.game.cursor_trail) == 0, f"Round {round_idx}: Lingering cursor trail"
            assert len(app.game.floating_texts) == 0, f"Round {round_idx}: Lingering floating texts"
            assert app.last_results_data == {}

            # 2. Simulate gameplay activity
            app.game.state = app.game.state.PLAYING  # skip countdown for test
            from src.game.fruit import Fruit, FruitType
            test_fruit = Fruit(x=640, y=360, vx=0, vy=0, fruit_data=FruitType.WATERMELON)
            app.game.fruits.append(test_fruit)
            app.game.update_cursor((0.48, 0.48), active_slice=True, slicing_mode="always")
            app.game.update_cursor((0.52, 0.52), active_slice=True, slicing_mode="always")
            app.game.update(0.1)

            assert app.game.score > 0
            assert app.game.fruits_sliced > 0

            # 3. Trigger Game Over
            app.game.trigger_game_over()
            app._on_game_over()

            assert app.screen_state == AppScreen.RESULTS
            assert app.last_results_data["score"] > 0
            assert app.game.high_score >= initial_high_score

        app._shutdown()

    def test_mandatory_e2e_sequence(self):
        """
        Executes the mandatory 14-step End-to-End sequence specified in Section 11:
        1. Open application.
        2. Start a game.
        3. Move hand and slice fruits.
        4. Pause the game.
        5. Resume the game.
        6. Trigger game-over.
        7. Click Play Again.
        8. Verify score, timer, fruits, and lives reset.
        9. Play a second complete round.
        10. Click Main Menu.
        11. Start another game.
        12. Open Settings and save a change.
        13. Exit the application.
        14. Relaunch and verify persistent high scores.
        """
        # Step 1: Open the application
        app = CyberSliceApp()
        assert app.screen_state == AppScreen.MAIN_MENU

        # Step 2: Start a game
        app.start_game()
        assert app.screen_state == AppScreen.PLAYING
        app.game.state = app.game.state.PLAYING  # advance past countdown

        # Step 3: Move the hand and slice several fruits
        from src.game.fruit import Fruit, FruitType
        f1 = Fruit(x=640, y=360, vx=0, vy=0, fruit_data=FruitType.ORANGE)
        f2 = Fruit(x=650, y=370, vx=0, vy=0, fruit_data=FruitType.APPLE)
        app.game.fruits.extend([f1, f2])
        app.game.update_cursor((0.50, 0.50), active_slice=True, slicing_mode="always")
        app.game.update_cursor((0.52, 0.52), active_slice=True, slicing_mode="always")
        assert app.game.fruits_sliced >= 1
        assert app.game.score > 0

        # Step 4: Pause the game
        app.game.toggle_pause()
        assert app.game.state == app.game.state.PAUSED
        paused_duration = app.game.game_duration
        app.game.update(0.5)
        # Verify timer progression stopped
        assert app.game.game_duration == paused_duration

        # Step 5: Resume the game
        app.game.toggle_pause()
        assert app.game.state == app.game.state.PLAYING

        # Step 6: Trigger game-over
        app.game.trigger_game_over()
        app._on_game_over()
        assert app.screen_state == AppScreen.RESULTS
        round1_score = app.last_results_data["score"]
        assert round1_score > 0

        # Step 7: Click Play Again
        app.results_screen.btn_play_again.trigger()

        # Step 8: Verify that score, timer, fruits, and lives reset
        assert app.screen_state == AppScreen.PLAYING
        assert app.game.score == 0
        assert app.game.game_duration == 0.0
        assert len(app.game.fruits) == 0
        assert app.game.lives == app.game.initial_lives
        assert app.game.fruits_sliced == 0
        assert app.game.bombs_hit == 0

        # Step 9: Play a second complete round
        app.game.state = app.game.state.PLAYING
        f3 = Fruit(x=640, y=360, vx=0, vy=0, fruit_data=FruitType.BANANA)
        app.game.fruits.append(f3)
        app.game.update_cursor((0.50, 0.50), active_slice=True, slicing_mode="always")
        app.game.trigger_game_over()
        app._on_game_over()
        assert app.screen_state == AppScreen.RESULTS

        # Step 10: Click Main Menu
        app.results_screen.btn_menu.trigger()
        assert app.screen_state == AppScreen.MAIN_MENU

        # Step 11: Start another game
        app.menu_screen.buttons[0].trigger()
        assert app.screen_state == AppScreen.PLAYING
        app.switch_screen(AppScreen.MAIN_MENU)

        # Step 12: Open Settings and save a change
        app.switch_screen(AppScreen.SETTINGS)
        assert app.screen_state == AppScreen.SETTINGS
        orig_diff = app.settings_screen.difficulty
        app.settings_screen.btn_diff.trigger()
        new_diff = app.settings_screen.difficulty
        assert orig_diff != new_diff
        app.settings_screen.btn_save_back.trigger()
        assert app.screen_state == AppScreen.MAIN_MENU
        assert app.game.difficulty == new_diff

        # Step 13: Exit the application
        app.exit_app()
        assert app.is_running is False
        app._shutdown()

        # Step 14: Relaunch and verify persistent high scores
        app2 = CyberSliceApp()
        assert app2.game.high_score >= round1_score
        app2._shutdown()

    def test_all_button_hitboxes_and_triggers(self):
        """Verifies every visible button across all screens has valid dimensions and functional callbacks."""
        app = CyberSliceApp()

        # Main Menu buttons
        for btn in app.menu_screen.buttons:
            assert btn.rect.width > 0 and btn.rect.height > 0
            assert 0 <= btn.rect.centerx <= 1280
            assert 0 <= btn.rect.centery <= 720
            assert btn.callback is not None

        # Settings buttons
        for btn in app.settings_screen.buttons:
            assert btn.rect.width > 0 and btn.rect.height > 0
            assert btn.callback is not None

        # Results buttons
        for btn in app.results_screen.buttons:
            assert btn.rect.width > 0 and btn.rect.height > 0
            assert btn.callback is not None

        # Game Pause buttons
        for btn in app.game_ui.pause_buttons:
            assert btn.rect.width > 0 and btn.rect.height > 0
            assert btn.callback is not None

        # How to play back button
        assert app.how_to_play_screen.btn_back.rect.width > 0
        assert app.how_to_play_screen.btn_back.callback is not None

        # Stats buttons
        for btn in app.stats_screen.buttons:
            assert btn.rect.width > 0 and btn.rect.height > 0
            assert btn.callback is not None

        # Practice buttons
        for btn in (app.practice_screen.back_button, app.practice_screen.reset_button):
            assert btn.rect.width > 0 and btn.rect.height > 0
            assert btn.callback is not None

        # Calibration buttons
        for btn in app.calibration_screen.buttons:
            assert btn.rect.width > 0 and btn.rect.height > 0
            assert btn.callback is not None

        app._shutdown()

