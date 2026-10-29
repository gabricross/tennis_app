from __future__ import annotations

import json
import uuid
from pathlib import Path

from playwright.sync_api import sync_playwright
from verification_config import parse_arguments


def main() -> None:
    settings = parse_arguments()
    PRIVATE, SOURCE, QUAD = settings.output, settings.video, settings.corners
    base_url = settings.base_url
    PRIVATE.mkdir(exist_ok=True)
    checks = {}
    errors = []
    external = []
    name = f"prueba-interfaz-{uuid.uuid4().hex[:6]}{SOURCE.suffix}"
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(channel="msedge", headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 1000}, device_scale_factor=1)
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.on(
            "request",
            lambda request: external.append(request.url)
            if not request.url.startswith((base_url + "/", "data:")) else None,
        )
        page.goto(base_url + "/", wait_until="networkidle")
        page.get_by_text("Motor de jugadores preparado", exact=True).wait_for()
        page.screenshot(path=str(PRIVATE / "app_desktop.png"), full_page=True)
        if page.evaluate("() => document.documentElement.scrollWidth>document.documentElement.clientWidth+1"):
            raise AssertionError("Horizontal overflow on desktop")
        checks["desktop_1440_no_overflow"] = True
        page.set_viewport_size({"width": 390, "height": 844})
        page.screenshot(path=str(PRIVATE / "app_mobile.png"), full_page=True)
        if page.evaluate("() => document.documentElement.scrollWidth>document.documentElement.clientWidth+1"):
            raise AssertionError("Horizontal overflow on mobile")
        checks["mobile_390_no_overflow"] = True
        page.set_viewport_size({"width": 1440, "height": 1000})
        with page.expect_response(lambda response: response.url.endswith("/api/clips") and response.request.method == "POST") as uploaded:
            page.locator("#file-input").set_input_files(
                {"name": name, "mimeType": "application/octet-stream", "buffer": SOURCE.read_bytes()}
            )
        response = uploaded.value
        if response.status != 201:
            raise AssertionError(f"Upload failed: HTTP {response.status}")
        page.locator("#configure-stage").wait_for(state="visible")
        page.wait_for_function("() => document.querySelector('#calibration-image').naturalWidth>0")
        library = page.request.get(base_url + "/api/library").json()
        clip = next((item for item in library["clips"] if item["name"] == name), None)
        if clip is None:
            raise AssertionError("The uploaded clip is missing from the library.")
        checks["file_uploaded_from_ui"] = True
        if settings.players_only:
            page.locator("#players-mode").check()
        if not settings.players_only:
            if not page.locator("#complete-mode").is_checked():
                raise AssertionError("A new clip must default to all three analysis layers.")
            if "PISTA + JUGADORES + PELOTA" not in page.locator("#analysis-plan").inner_text():
                raise AssertionError("The complete analysis must explicitly include all three layers.")
            checks["combined_analysis_is_explicit_default"] = True
            page.locator("#players-mode").check()
            if page.locator("#ball-options").is_visible() or "NO está incluida" not in page.locator("#analysis-plan").inner_text():
                raise AssertionError("Players-only mode must explicitly exclude the ball.")
            checks["players_only_mode_is_explicit"] = True
        page.get_by_label("Duración del análisis", exact=True).fill("2")
        if not settings.players_only:
            page.locator("#complete-mode").check()
            page.locator("#ball-backend").select_option(settings.ball_backend)
            checks["ball_backend_selected_in_ui"] = True
        if settings.start:
            page.get_by_label("Inicio, en segundos", exact=True).fill(str(settings.start))
            page.get_by_role("button", name="Actualizar fotograma", exact=True).click()
            page.get_by_role("button", name="Actualizar fotograma", exact=True).wait_for(state="visible")
            page.wait_for_function("() => !document.querySelector('#load-frame').disabled")
        if settings.reference != "doubles":
            page.get_by_label("Referencias visibles para calibrar", exact=True).select_option(settings.reference)
            if "NO marques la cinta de la red" not in page.locator("#reference-help").inner_text():
                raise AssertionError("The service reference needs an explicit ground-plane guide.")
            checks["partial_calibration_guide_visible"] = True
        elif settings.players_only:
            if page.get_by_role("button", name="Proponer pista con IA", exact=True).is_enabled():
                raise AssertionError("Court proposal must be unavailable without its model.")
            if page.locator("#complete-mode").is_enabled():
                raise AssertionError("Ball mode must be unavailable without its model.")
            checks["missing_optional_models_are_explicit"] = True
        else:
            page.get_by_role("button", name="Proponer pista con IA", exact=True).click()
            page.wait_for_function("() => document.querySelectorAll('#corner-fields .corner-row').length===4")
            checks["court_model_is_connected_to_ui"] = True
        if page.locator("#start-analysis").is_enabled():
            raise AssertionError("The analysis must require explicit court review.")
        checks["explicit_court_confirmation_required"] = True
        page.get_by_role("button", name="Volver a marcar", exact=True).click()
        canvas = page.locator("#calibration-canvas")
        for x, y in QUAD:
            bounds = canvas.bounding_box()
            canvas.click(position={"x": x * bounds["width"], "y": y * bounds["height"]})
        page.locator("#confirm-court").check()
        if not page.locator("#start-analysis").is_enabled():
            raise AssertionError("A reviewed manual calibration should enable analysis.")
        checks["manual_corners_work"] = True
        if "NO mide precisión" not in page.locator("#calibration-quality").inner_text():
            raise AssertionError("Calibration support must not be presented as accuracy.")
        if page.locator("#reference-diagram").evaluate("(canvas)=>canvas.width") != 260:
            raise AssertionError("The numbered reference diagram is missing.")
        checks["complete_court_projection_reviewed"] = True
        if settings.reference != "doubles":
            page.get_by_label("Referencias visibles para calibrar", exact=True).select_option("doubles")
            if page.locator("#corner-fields .corner-row").count() != 0 or page.locator("#confirm-court").is_checked():
                raise AssertionError("Changing reference must clear the old points and review.")
            if page.locator("#start-analysis").is_enabled():
                raise AssertionError("Old coordinates must not enable a different reference.")
            page.get_by_label("Referencias visibles para calibrar", exact=True).select_option(settings.reference)
            for x, y in QUAD:
                bounds = canvas.bounding_box()
                canvas.click(position={"x": x * bounds["width"], "y": y * bounds["height"]})
            page.locator("#confirm-court").check()
            checks["changing_reference_invalidates_old_points"] = True
        page.get_by_label("Inicio, en segundos", exact=True).fill(str(settings.start + 1))
        if page.locator("#start-analysis").is_enabled():
            raise AssertionError("Changing the start must invalidate the frame confirmation.")
        page.get_by_label("Inicio, en segundos", exact=True).fill(str(settings.start))
        checks["changed_start_requires_matching_frame"] = True
        page.set_viewport_size({"width": 390, "height": 844})
        if page.evaluate("()=>document.documentElement.scrollWidth>document.documentElement.clientWidth+1"):
            raise AssertionError("The configured ball selector has horizontal overflow on mobile.")
        checks["mobile_configuration_no_overflow"] = True
        page.set_viewport_size({"width": 1440, "height": 1000})
        page.screenshot(path=str(PRIVATE / "app_calibration.png"), full_page=True)
        with page.expect_response(lambda response: response.url.endswith("/api/jobs") and response.request.method == "POST") as started:
            page.locator("#start-analysis").click()
        job = started.value.json()
        if job["config"]["calibration_reference"] != settings.reference:
            raise AssertionError("The API did not preserve the selected calibration reference.")
        checks["calibration_reference_persisted"] = True
        if not settings.players_only:
            if job["config"]["ball_backend"] != settings.ball_backend or job["config"]["mode"] != "complete":
                raise AssertionError("The UI did not submit the selected ball backend.")
            checks["ball_backend_persisted"] = True
        page.locator("#progress-stage").wait_for(state="visible")
        checks["real_progress_visible"] = True
        page.locator("#results-stage").wait_for(state="visible", timeout=180000)
        checks["upload_to_results_complete"] = True
        if "Distancia observada:" not in page.locator("#south-motion").inner_text():
            raise AssertionError("The UI must expose observed player distance and estimated speed.")
        if "km/h" not in page.locator("#south-current-speed").inner_text() and "no disponible" not in page.locator("#south-current-speed").inner_text():
            raise AssertionError("Current speed must be explicitly estimated or unavailable.")
        checks["estimated_player_movement_visible"] = True
        page.wait_for_function("() => document.querySelector('#result-video').readyState>=2")
        page.locator("#result-seek").evaluate(
            "el=>{el.value='0.8';el.dispatchEvent(new Event('input',{bubbles:true}));}"
        )
        page.wait_for_function("()=>Math.abs(document.querySelector('#result-video').currentTime-0.8)<0.1")
        page.get_by_role("button", name="Original", exact=True).click()
        page.wait_for_function(
            "()=>document.querySelector('#result-video').readyState>=2"
            "&&document.querySelector('#result-video').currentSrc.endsWith('/original.mp4')"
            "&&Math.abs(document.querySelector('#result-video').currentTime-0.8)<0.15"
        )
        page.get_by_role("button", name="Analizado", exact=True).click()
        page.wait_for_function(
            "()=>document.querySelector('#result-video').readyState>=2"
            "&&document.querySelector('#result-video').currentSrc.endsWith('/analysis.mp4')"
        )
        checks["original_analysed_switch_preserves_time"] = True
        page.locator("#result-video").evaluate("video=>video.play()")
        page.wait_for_function("()=>document.querySelector('#result-video').currentTime>1")
        page.locator("#result-video").evaluate("video=>video.pause()")
        checks["native_video_actually_plays"] = True
        positions = page.locator("#result-map").evaluate("canvas=>canvas.toDataURL()")
        page.get_by_role("button", name="Densidad", exact=True).click()
        density = page.locator("#result-map").evaluate("canvas=>canvas.toDataURL()")
        if density == positions:
            raise AssertionError("The density control did not change the observations map.")
        checks["density_control_works"] = True
        page.locator("#player-filter").select_option("north")
        north_only = page.locator("#result-map").evaluate("canvas=>canvas.toDataURL()")
        if density == north_only:
            raise AssertionError("The player filter did not change the map.")
        checks["player_filter_works"] = True
        with page.expect_popup() as popup:
            page.get_by_role("link", name="Comparar ambos", exact=False).click()
        comparison = popup.value
        comparison.on("pageerror", lambda error: errors.append(str(error)))
        comparison.locator("#play").wait_for(state="visible")
        comparison.wait_for_function(
            "()=>document.querySelectorAll('video').length===2"
            "&&[...document.querySelectorAll('video')].every(v=>v.readyState>=2)"
            "&&!document.querySelector('#play').disabled"
        )
        comparison.get_by_role("button", name="Reproducir ambos", exact=True).click()
        comparison.wait_for_function(
            "()=>[...document.querySelectorAll('video')].every(v=>v.currentTime>0.3)"
        )
        comparison.get_by_role("button", name="Pausar", exact=True).click()
        drift = comparison.evaluate(
            "()=>Math.abs(document.querySelector('#original').currentTime-document.querySelector('#analysis').currentTime)"
        )
        if drift > 0.25 or comparison.locator("#error").is_visible():
            raise AssertionError("The comparison videos did not play together.")
        if "m observados" not in comparison.locator("#movement-note").inner_text():
            raise AssertionError("The comparison must include observed movement estimates.")
        checks["comparison_exposes_player_movement"] = True
        if not settings.players_only:
            if not comparison.locator("#ball-card").is_visible():
                raise AssertionError("The comparison must show per-frame ball coverage.")
            if not comparison.locator("#ball-csv").get_attribute("href").endswith("/ball.csv"):
                raise AssertionError("The comparison must expose the actual ball export.")
            checks["comparison_exposes_ball_coverage_and_csv"] = True
        comparison.set_viewport_size({"width": 390, "height": 844})
        if comparison.evaluate("()=>document.documentElement.scrollWidth>document.documentElement.clientWidth+1"):
            raise AssertionError("The comparison has horizontal overflow on mobile.")
        checks["mobile_comparison_no_overflow"] = True
        comparison.close()
        checks["side_by_side_comparison_actually_plays"] = True
        with page.expect_download() as downloaded:
            page.locator('#export-links a[href$="/observations.csv"]').click()
        download = downloaded.value
        destination = PRIVATE / "ui_download.csv"
        download.save_as(destination)
        if "player_side" not in destination.read_text(encoding="utf-8-sig"):
            raise AssertionError("The downloaded CSV is not the real observation export.")
        checks["download_contains_real_csv"] = True
        if not settings.players_only:
            with page.expect_download() as downloaded:
                page.locator('#export-links a[href$="/ball.csv"]').click()
            destination = PRIVATE / "ui_ball_download.csv"
            downloaded.value.save_as(destination)
            if "detector_score" not in destination.read_text(encoding="utf-8-sig"):
                raise AssertionError("The ball CSV must contain the actual detector output fields.")
            checks["ball_download_contains_real_csv"] = True
        page.get_by_role("button", name="Posiciones", exact=True).click()
        page.evaluate("() => window.scrollTo(0,0)")
        page.screenshot(path=str(PRIVATE / "app_results.png"), full_page=True)
        page.set_viewport_size({"width": 390, "height": 844})
        if page.evaluate("()=>document.documentElement.scrollWidth>document.documentElement.clientWidth+1"):
            raise AssertionError("Results have horizontal overflow on mobile.")
        checks["mobile_results_no_overflow"] = True
        page.screenshot(path=str(PRIVATE / "app_results_mobile.png"), full_page=True)
        page.set_viewport_size({"width": 1440, "height": 1000})
        for _ in range(5):
            page.get_by_role("button", name="Biblioteca", exact=False).first.click()
            card = page.locator(".library-card").filter(has_text=name)
            card.get_by_role("button", name="Analizar de nuevo", exact=True).click()
            page.wait_for_function("()=>document.querySelectorAll('#corner-fields .corner-row').length===4")
            page.locator("#confirm-court").check()
            page.locator("#start-analysis").click()
            page.locator("#progress-stage").wait_for(state="visible")
            page.get_by_role("button", name="Cancelar análisis", exact=True).click()
            page.get_by_role("heading", name="Análisis cancelado.", exact=True).wait_for(timeout=30000)
        checks["cancel_from_ui_works"] = True
        page.get_by_role("button", name="Biblioteca", exact=False).first.click()
        card = page.locator(".library-card").filter(has_text=name)
        card.get_by_role("button", name="Ver los 6 análisis", exact=True).click()
        card.get_by_role("button", name="Terminado", exact=False).click()
        page.locator("#results-stage").wait_for(state="visible")
        checks["older_results_remain_accessible"] = True
        checks["history_reopens_persisted_results"] = True
        page.get_by_role("button", name="Biblioteca", exact=False).first.click()
        card = page.locator(".library-card").filter(has_text=name)
        page.once("dialog", lambda dialog: dialog.accept())
        card.get_by_role("button", name="Borrar clip y resultados", exact=True).click()
        page.wait_for_function(
            "(name)=>![...document.querySelectorAll('.library-card')].some(n=>n.textContent.includes(name))",
            arg=name,
        )
        removed = page.request.get(f"{base_url}/api/clips/{clip['id']}/poster")
        if removed.status != 404:
            raise AssertionError("Deleting the UI fixture did not remove the stored clip.")
        checks["clip_and_results_delete_from_ui"] = True
        page.get_by_role("button", name="Motor y privacidad", exact=False).click()
        if not page.locator("#storage-path").inner_text().strip():
            raise AssertionError("Private storage location is not visible.")
        checks["local_storage_location_visible"] = True
        browser.close()
    if errors:
        raise AssertionError(f"Unhandled JavaScript errors: {errors}")
    if external:
        raise AssertionError(f"The app contacted external origins: {external}")
    checks["no_unhandled_javascript_errors"] = True
    checks["no_external_app_requests"] = True
    report = {"checks": checks, "count": len(checks), "javascript_errors": errors,
              "external_requests": external, "test_job": job["id"],
              "notice": "Application workflow test on a local user-provided clip; not a model accuracy benchmark."}
    (PRIVATE / "browser_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
