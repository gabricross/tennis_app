from __future__ import annotations

import argparse
import re
import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from playwright.sync_api import expect, sync_playwright

ROOT = Path(__file__).resolve().parents[1]


def verify_pair(page, pair_index: int = 0) -> None:
    controls = page.locator("[data-video-pair]").nth(pair_index)
    ids = controls.get_attribute("data-video-pair").split()
    play = controls.locator("[data-video-action=play]")
    pause = controls.locator("[data-video-action=pause]")
    restart = controls.locator("[data-video-action=restart]")
    state = """(ids) => {
        const [a,b]=ids.map(id=>document.getElementById(id));
        return {times:[a.currentTime,b.currentTime],paused:[a.paused,b.paused],delta:Math.abs(a.currentTime-b.currentTime)};
    }"""
    expect(play).to_be_enabled(timeout=30000)
    page.locator("#" + ids[0]).evaluate("(v)=>{v.currentTime=2;}")
    page.wait_for_function("(ids)=>ids.every(id=>Math.abs(document.getElementById(id).currentTime-2)<0.15)", arg=ids)
    page.locator("#" + ids[1]).evaluate("(v)=>{v.currentTime=6;}")
    play.click()
    page.wait_for_function("(ids)=>ids.every(id=>document.getElementById(id).currentTime>2.3)", arg=ids)
    result = page.evaluate(state, ids)
    assert result["delta"] < 0.18 and not any(result["paused"]), result
    pause.click()
    assert all(page.evaluate(state, ids)["paused"])
    restart.click()
    page.wait_for_function("(ids)=>ids.every(id=>document.getElementById(id).currentTime===0)", arg=ids)
    assert all(page.evaluate(state, ids)["paused"])

    play.click()
    page.wait_for_function("(ids)=>ids.every(id=>document.getElementById(id).currentTime>0.3)", arg=ids)
    page.locator("#" + ids[1]).evaluate("(v)=>{v.currentTime=8;}")
    page.wait_for_function("(ids)=>Math.abs(document.getElementById(ids[0]).currentTime-document.getElementById(ids[1]).currentTime)<0.18", arg=ids)
    page.locator("#" + ids[0]).evaluate("(v)=>{v.playbackRate=1.5;}")
    page.wait_for_function("(ids)=>ids.every(id=>document.getElementById(id).playbackRate===1.5)", arg=ids)
    page.locator("#" + ids[1]).evaluate("(v)=>v.pause()")
    page.wait_for_function("(ids)=>ids.every(id=>document.getElementById(id).paused)", arg=ids)
    page.locator("#" + ids[0]).evaluate("(v)=>{v.playbackRate=1;}")
    page.locator("#" + ids[0]).evaluate("(v)=>{v.currentTime=v.duration-0.2;}")
    play.click()
    page.wait_for_function("(ids)=>ids.every(id=>{const v=document.getElementById(id);return v.paused&&v.currentTime>v.duration-0.3;})", arg=ids)
    play.click()
    page.wait_for_function("(ids)=>ids.every(id=>{const v=document.getElementById(id);return !v.paused&&v.currentTime>0.2&&v.currentTime<5;})", arg=ids)
    pause.click()

    page.locator("#" + ids[1]).evaluate("""(v)=>{
        const play=v.play;
        window.restorePairPlay=()=>{v.play=play;};
        v.play=()=>Promise.reject(new Error("prueba de reproducción bloqueada"));
    }""")
    play.click()
    expect(controls.locator("[data-pair-error]")).to_contain_text("prueba de reproducción bloqueada")
    assert all(page.evaluate(state, ids)["paused"])
    page.evaluate("() => window.restorePairPlay()")
    play.click()
    page.wait_for_function("(ids)=>ids.every(id=>!document.getElementById(id).paused)", arg=ids)
    expect(controls.locator("[data-pair-error]")).to_be_hidden()
    page.locator("#" + ids[0]).evaluate("(v)=>v.dispatchEvent(new Event('error'))")
    expect(play).to_be_disabled()
    assert all(page.evaluate(state, ids)["paused"])
    page.locator("#" + ids[0]).locator("..").get_by_role("button", name="Reintentar").click()
    expect(play).to_be_enabled(timeout=30000)
    restart.click()
    page.wait_for_function("(ids)=>ids.every(id=>document.getElementById(id).currentTime===0)", arg=ids)
    print("PASS: paired play (<0.18 s drift), pause, seek, rate, restart, replay after end and failure recovery.")


def verify_examples(page) -> None:
    examples = page.locator(".media-example")
    expect(examples).to_have_count(2)
    first, second = examples.nth(0).bounding_box(), examples.nth(1).bounding_box()
    assert second["y"] >= first["y"] + first["height"], (first, second)
    expect(examples.nth(0).locator("h2")).to_contain_text("iPhone")
    expect(examples.nth(1).locator("h2")).to_contain_text("aérea")
    expect(examples.nth(1)).to_contain_text("7,2 %")
    for index in range(2):
        verify_pair(page, index)
    pairs = [example.locator("video") for example in (examples.nth(0), examples.nth(1))]
    examples.nth(0).locator("[data-video-action=play]").click()
    page.wait_for_function("() => [...document.querySelectorAll('.media-example:first-child video')].every(v=>!v.paused&&v.currentTime>0.2)")
    assert pairs[1].evaluate_all("(vs)=>vs.every(v=>v.paused&&v.currentTime===0)")
    examples.nth(1).locator("[data-video-action=play]").click()
    page.wait_for_function("() => [...document.querySelectorAll('.media-example:last-child video')].every(v=>!v.paused&&v.currentTime>0.2)")
    examples.nth(0).locator("[data-video-action=pause]").click()
    assert pairs[0].evaluate_all("(vs)=>vs.every(v=>v.paused)")
    assert pairs[1].evaluate_all("(vs)=>vs.every(v=>!v.paused)")
    examples.nth(0).locator("[data-video-action=restart]").click()
    assert pairs[1].evaluate_all("(vs)=>vs.every(v=>!v.paused)")
    examples.nth(1).locator("[data-video-action=pause]").click()
    print("PASS: two vertically stacked examples, with independent loading and playback controls.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify full-download video loading with Edge.")
    parser.add_argument("--base-url", help="Also check a running local Tennis Lab instance.")
    arguments = parser.parse_args()
    media = (ROOT / "docs" / "media" / "caltennis_original_20s.mp4").read_bytes()
    started, release = threading.Event(), threading.Event()
    attempts = 0

    class Handler(SimpleHTTPRequestHandler):
        def log_message(self, format, *args):
            pass

        def do_GET(self):
            nonlocal attempts
            if self.path == "/fixture":
                body = (
                    '<meta charset="utf-8"><link rel="stylesheet" href="/assets/video-loader.css">'
                    '<video hidden playsinline preload="none"></video>'
                    '<script src="/assets/video-loader.js"></script>'
                    '<script>window.loader=new TennisVideoLoader(document.querySelector("video"));</script>'
                ).encode()
            elif self.path in ("/slow.mp4", "/retry.mp4", "/broken.mp4", "/empty.mp4"):
                if self.path == "/retry.mp4":
                    attempts += 1
                    if attempts == 1:
                        self.send_error(503)
                        return
                body = b"invalid MP4" if self.path == "/broken.mp4" else b"" if self.path == "/empty.mp4" else media
            else:
                super().do_GET()
                return
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Content-Type", "text/html" if self.path == "/fixture" else "video/mp4")
            self.end_headers()
            if self.path == "/slow.mp4":
                self.wfile.write(body[:len(body) // 2])
                self.wfile.flush()
                started.set()
                if not release.wait(20):
                    raise TimeoutError("The test did not release its slow video response.")
                try:
                    self.wfile.write(body[len(body) // 2:])
                except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                    # Source-switch tests intentionally cancel the old download.
                    return
            else:
                self.wfile.write(body)

    server = ThreadingHTTPServer(("127.0.0.1", 0), partial(Handler, directory=ROOT / "docs"))
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(channel="msedge", headless=True)
            page = browser.new_page(viewport={"width": 1280, "height": 900})
            errors = []
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.goto(base + "/fixture")
            page.evaluate("""() => {
                window.revoked=[];
                const revoke=URL.revokeObjectURL.bind(URL);
                URL.revokeObjectURL=url=>{revoked.push(url);revoke(url);};
                window.done=null;
                void loader.load("/slow.mp4").then(value=>{window.done=value;});
            }""")
            assert started.wait(10)
            page.wait_for_function("() => document.querySelector('progress').value>0")
            expect(page.locator("video")).to_be_hidden()
            assert page.evaluate("() => !document.querySelector('video').controls && !document.querySelector('video').hasAttribute('src')")
            assert page.evaluate("() => done===null && document.querySelector('progress').value<100")
            expect(page.get_by_role("status")).to_have_text("Cargando vídeo…")
            release.set()
            page.wait_for_function("() => window.done===true")
            expect(page.locator("video")).to_be_visible()
            assert page.evaluate("() => loader.ready && document.querySelector('video').duration===20")
            page.evaluate("() => document.querySelector('video').play()")
            page.wait_for_function("() => document.querySelector('video').currentTime>0.4")
            page.evaluate("() => document.querySelector('video').pause()")
            page.evaluate("() => document.querySelector('video').dispatchEvent(new Event('error'))")
            expect(page.get_by_role("alert")).to_contain_text("continuar la reproducción")
            page.get_by_role("button", name="Reintentar").click()
            page.wait_for_function("() => loader.ready")
            print("PASS: hidden until every byte arrives; then visible and playing.")

            old_url = page.locator("video").get_attribute("src")
            assert page.evaluate("() => loader.load('/media/caltennis_original_20s.mp4',{time:5})")
            assert page.evaluate("() => Math.abs(document.querySelector('video').currentTime-5)<0.1")
            assert page.evaluate("(url) => revoked.includes(url)", old_url)
            for path, message in [("/missing.mp4", "HTTP 404"), ("/broken.mp4", "no puede reproducir"), ("/empty.mp4", "vacío")]:
                assert page.evaluate("(path) => loader.load(path)", path) is False
                expect(page.locator("video")).to_be_hidden()
                expect(page.get_by_role("alert")).to_contain_text(message)
                expect(page.get_by_role("button", name="Reintentar")).to_be_visible()
            page.route("**/offline.mp4", lambda route: route.abort())
            assert page.evaluate("() => loader.load('/offline.mp4')") is False
            expect(page.get_by_role("alert")).to_contain_text("No hay conexión")
            assert page.evaluate("() => loader.load('/retry.mp4')") is False
            page.get_by_role("button", name="Reintentar").click()
            page.wait_for_function("() => loader.ready")
            assert attempts == 2
            print("PASS: source time restored, object URLs revoked, HTTP/network/empty/decode errors and retry.")

            started.clear()
            release.clear()
            page.evaluate("""() => {
                window.oldDone=null;
                void loader.load('/slow.mp4').then(value=>{window.oldDone=value;});
            }""")
            assert started.wait(10)
            assert page.evaluate("() => loader.load('/media/caltennis_original_20s.mp4',{time:7})")
            release.set()
            page.wait_for_function("() => oldDone===false")
            assert page.evaluate("() => loader.ready && Math.abs(document.querySelector('video').currentTime-7)<0.1")
            page.evaluate("() => loader.clear()")
            assert page.evaluate("() => !loader.ready && !document.querySelector('video').hasAttribute('src')")
            print("PASS: rapid source replacement and clearing cannot reveal stale downloads.")

            started.clear()
            release.clear()
            page.route("**/media/caltennis_original_20s.mp4", lambda route: route.continue_(url=base + "/slow.mp4"))
            page.goto(base + "/index.html")
            page.wait_for_function("() => document.querySelector('.media-example .video-loader[data-state=loading] progress')?.value>0")
            assert started.wait(10)
            page.wait_for_function("() => document.querySelector('.media-example .video-loader[data-state=ready]')")
            for action in ("play", "pause", "restart"):
                expect(page.locator(".media-example").first.locator(f"[data-video-action={action}]")).to_be_disabled()
            expect(page.locator(".media-example").nth(1).locator("[data-video-action=play]")).to_be_enabled(timeout=30000)
            release.set()
            page.unroute("**/media/caltennis_original_20s.mp4")
            page.wait_for_function("() => document.querySelectorAll('.video-loader[data-state=ready]').length===4")
            for width in (1280, 390):
                page.set_viewport_size({"width": width, "height": 900})
                assert page.evaluate("() => document.documentElement.scrollWidth<=document.documentElement.clientWidth+1")
            assert page.evaluate("() => [...document.querySelectorAll('video')].every(v=>v.duration>=20&&v.duration<23&&v.currentSrc.startsWith('blob:')&&!v.error)")
            print("PASS: public page has four fully loaded videos without desktop/mobile overflow.")
            verify_examples(page)

            if arguments.base_url:
                app_url = arguments.base_url.rstrip("/")
                page.goto(app_url + "/")
                page.wait_for_function("() => document.querySelectorAll('#showcase-examples .video-loader[data-state=ready]').length===4")
                verify_examples(page)
                assert page.locator("#use-previous").count() == 0
                config = page.request.get(app_url + "/api/config").json()
                library = page.request.get(app_url + "/api/library").json()
                job_id = config["showcase"]["job_id"]
                job = next(item for item in library["jobs"] if item["id"] == job_id)
                clip = next(item for item in library["clips"] if item["id"] == job["clip_id"])
                jobs = [item for item in library["jobs"] if item["clip_id"] == clip["id"]]
                index = next(i for i, item in enumerate(jobs) if item["id"] == job_id)
                page.get_by_role("button", name=re.compile("^Biblioteca")).click()
                card = page.locator(".library-card").filter(has=page.get_by_role("heading", name=clip["name"], exact=True))
                if index >= 4:
                    card.get_by_role("button", name=re.compile("^Ver los")).click()
                card.locator(".library-actions button").nth(index + 1).click()
                page.wait_for_function("() => document.querySelector('#result-video').parentElement.dataset.state==='ready'")
                expect(page.locator("#result-seek")).to_be_enabled()
                page.locator("#result-seek").evaluate("(e)=>{e.value=5;e.dispatchEvent(new Event('input',{bubbles:true}));}")
                page.locator("[data-result-view=original]").click()
                page.wait_for_function("() => document.querySelector('#result-video').parentElement.dataset.state==='ready'")
                assert abs(page.locator("#result-video").evaluate("(v)=>v.currentTime") - 5) < 0.1
                page.goto(app_url + f"/api/jobs/{job_id}/comparison")
                verify_pair(page)
                print("PASS: local showcase, result switching/seek and dual comparison playback.")
            assert not errors, errors
            browser.close()
    finally:
        release.set()
        server.shutdown()
        server.server_close()
        worker.join()


if __name__ == "__main__":
    main()
