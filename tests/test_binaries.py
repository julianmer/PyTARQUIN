####################################################################################################
#                                        test_binaries.py                                          #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 2026-10-01                                                                              #
#                                                                                                  #
# Purpose: Tests for TARQUIN executable resolution - the health probe, platform detection, the     #
#          release download and the container launcher.                                            #
#                                                                                                  #
#          These need no TARQUIN binary and no network. The tests that do use a real binary        #
#          skip themselves when none is resolvable.                                                #
#                                                                                                  #
####################################################################################################

import hashlib
import lzma
import os
import platform
import shutil
import subprocess
import sys

import pytest

_REPO = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, _REPO)

from tarquin_wrapper import __version__, binaries, container


#*************#
#   helpers   #
#*************#
def _make_elf(path, e_machine):
    """Write a minimal ELF64 header with the given e_machine, marked executable."""
    hdr = bytearray(64)
    hdr[0:4] = b"\x7fELF"
    hdr[4], hdr[5], hdr[6] = 2, 1, 1          # 64-bit, little endian, version 1
    hdr[16:18] = (2).to_bytes(2, "little")     # ET_EXEC
    hdr[18:20] = e_machine.to_bytes(2, "little")
    path.write_bytes(bytes(hdr))
    os.chmod(path, 0o755)
    return path


def _real_binary():
    """TARQUIN_EXEC, or a cached download, or None."""
    if os.environ.get("TARQUIN_EXEC"):
        return os.environ["TARQUIN_EXEC"]
    path = binaries._cache_dir() / binaries._exec_name()
    return str(path) if path.is_file() else None


def _fake_engine(tmp_path, body):
    """A stand-in for the docker CLI: a script that records its arguments."""
    if os.name == "nt":
        pytest.skip("POSIX shell script")
    engine = tmp_path / "fake-docker"
    engine.write_text("#!/bin/sh\n" + body)
    os.chmod(engine, 0o755)
    return engine


#**********************#
#   the health probe   #
#**********************#
def test_probe_rejects_a_program_that_does_not_identify_itself():
    """TARQUIN ends '--help' with 255 and may exit 0 on a fit it gave up on, so the exit
    status says nothing; its start line is the only signal."""
    ok, why = binaries.verify_executable(sys.executable)
    assert ok is False
    assert "did not identify itself as TARQUIN" in why


def test_probe_rejects_wrong_architecture(tmp_path):
    """A truncated ELF stands in for a wrong-architecture download: neither runs, and the
    message names what the file is."""
    ok, why = binaries.verify_executable(_make_elf(tmp_path / "tarquin", 0x3E))
    assert ok is False
    # refused by the kernel, or - where an emulator is registered for x86-64 - killed
    assert why.startswith(("cannot execute", "killed by signal"))
    assert "x86-64" in why          # names the mismatch instead of "Exec format error"


def test_probe_describes_aarch64_too(tmp_path):
    ok, why = binaries.verify_executable(_make_elf(tmp_path / "tarquin", 0xB7))
    assert ok is False
    assert "AArch64" in why


def test_probe_rejects_non_executable_content(tmp_path):
    path = tmp_path / "tarquin"
    path.write_text("this is not a binary\n")
    os.chmod(path, 0o755)
    ok, _ = binaries.verify_executable(path)
    assert ok is False


def test_probe_times_out_instead_of_hanging(tmp_path):
    if os.name == "nt":
        pytest.skip("POSIX shell script")
    path = tmp_path / "slowpoke"
    path.write_text("#!/bin/sh\nsleep 30\n")
    os.chmod(path, 0o755)
    ok, why = binaries.verify_executable(path, timeout=1.0)
    assert ok is False
    assert "did not exit" in why


def test_probe_reports_missing_file(tmp_path):
    ok, why = binaries.verify_executable(tmp_path / "does-not-exist")
    assert ok is False
    assert "cannot stat" in why


def test_probe_memoizes(tmp_path):
    path = _make_elf(tmp_path / "tarquin", 0x3E)
    first = binaries.verify_executable(path)
    assert binaries.verify_executable(path) == first
    st = path.stat()
    assert (str(path.resolve()), st.st_mtime_ns, st.st_size) in binaries._VERIFIED


@pytest.mark.skipif(_real_binary() is None, reason="no TARQUIN binary")
def test_probe_accepts_the_real_binary():
    ok, why = binaries.verify_executable(_real_binary())
    assert ok is True, why


@pytest.mark.skipif(_real_binary() is None, reason="no TARQUIN binary")
def test_probe_accepts_a_relative_path(tmp_path, monkeypatch):
    """The probe runs the candidate in a temporary directory, so a path relative to the
    caller's directory (a relative TARQUIN_EXEC, say) has to be resolved first."""
    (tmp_path / "bin").mkdir()
    shutil.copy2(_real_binary(), tmp_path / "bin" / binaries._exec_name())
    monkeypatch.chdir(tmp_path)
    ok, why = binaries.verify_executable(os.path.join("bin", binaries._exec_name()))
    assert ok is True, why


#************************#
#   platform detection   #
#************************#
@pytest.mark.parametrize("system, machine, expected", [
    ("Linux",   "x86_64",  ["linux-x86_64"]),
    ("Linux",   "AMD64",   ["linux-x86_64"]),
    ("Linux",   "aarch64", ["linux-aarch64"]),
    ("Linux",   "arm64",   ["linux-aarch64"]),
    ("Linux",   "armv7l",  []),                 # falls through to the container
    ("Darwin",  "arm64",   ["darwin-arm64", "darwin-x86_64"]),
    ("Darwin",  "x86_64",  ["darwin-x86_64"]),
    ("Windows", "AMD64",   ["windows-amd64"]),
    ("Windows", "ARM64",   ["windows-amd64"]),  # emulates x64
    ("FreeBSD", "amd64",   []),
])
def test_platform_keys(monkeypatch, system, machine, expected):
    monkeypatch.setattr(platform, "system", lambda: system)
    monkeypatch.setattr(platform, "machine", lambda: machine)
    assert binaries._platform_keys() == expected


def test_every_release_asset_is_reachable(monkeypatch):
    reachable = set()
    for system, machine in [("Linux", "x86_64"), ("Linux", "aarch64"), ("Darwin", "arm64"),
                            ("Darwin", "x86_64"), ("Windows", "AMD64")]:
        monkeypatch.setattr(platform, "system", lambda s=system: s)
        monkeypatch.setattr(platform, "machine", lambda m=machine: m)
        reachable.update(binaries._platform_keys())
    assert set(binaries._RELEASE_ASSETS) == reachable


#*************#
#   release   #
#*************#
@pytest.mark.parametrize("system, machine, expected", [
    ("Linux",   "x86_64",  ["tarquin-linux-x86_64.xz"]),
    ("Linux",   "aarch64", ["tarquin-linux-aarch64.xz"]),
    ("Darwin",  "arm64",   ["tarquin-macos-arm64.xz", "tarquin-macos-x86_64.xz"]),
    ("Darwin",  "x86_64",  ["tarquin-macos-x86_64.xz"]),
    ("Windows", "AMD64",   ["tarquin-windows-x86_64.xz"]),
    ("FreeBSD", "amd64",   []),
])
def test_release_assets_per_platform(monkeypatch, system, machine, expected):
    """Apple silicon lists the Intel asset too, as a last resort through Rosetta."""
    monkeypatch.setattr(platform, "system", lambda: system)
    monkeypatch.setattr(platform, "machine", lambda: machine)
    assert binaries._release_assets() == expected


def test_release_tag_follows_the_package_version(monkeypatch):
    """A wheel pulls the binaries built alongside it: release and image are tagged
    "v<version>", the same tag that marks the PyPI release."""
    monkeypatch.delenv("TARQUIN_RELEASE_TAG", raising=False)
    monkeypatch.delenv("TARQUIN_DOCKER_IMAGE", raising=False)
    assert binaries._release_tag() == f"v{__version__}"
    assert binaries._container_image() == f"ghcr.io/julianmer/tarquin:v{__version__}"


def test_release_tag_is_overridable(monkeypatch):
    monkeypatch.setenv("TARQUIN_RELEASE_TAG", "v9.9.9")
    assert binaries._release_base().endswith("/releases/download/v9.9.9")
    assert binaries._container_image().endswith(":v9.9.9")
    monkeypatch.setenv("TARQUIN_DOCKER_IMAGE", "example.org/me/tarquin:dev")
    assert binaries._container_image() == "example.org/me/tarquin:dev"


def _serve(monkeypatch, payload, digest):
    """Make the release download hand out "payload" (xz-compressed) and "digest"."""
    def fake_download(url, dest):
        if url.endswith(".sha256"):
            dest.write_text(f"{digest}  tarquin-linux-x86_64.xz\n")
        else:
            dest.write_bytes(payload)
    monkeypatch.setattr(binaries, "_download", fake_download)
    monkeypatch.setattr(binaries, "_release_assets", lambda: ["tarquin-linux-x86_64.xz"])


_FAKE = b"#!/bin/sh\necho 'TARQUIN 4.3.11 Started'\n"


def test_release_download_installs_a_verified_binary(monkeypatch, tmp_path):
    if os.name == "nt":
        pytest.skip("POSIX shell script")
    payload = lzma.compress(_FAKE)
    _serve(monkeypatch, payload, hashlib.sha256(payload).hexdigest())
    got = binaries._download_release_binary(tmp_path)
    assert got == tmp_path / binaries._exec_name()
    assert got.read_bytes() == _FAKE
    assert os.access(got, os.X_OK)


def test_a_release_binary_that_does_not_run_gives_the_next_its_turn(monkeypatch, tmp_path):
    """Apple silicon lists the Intel build after its own: one that downloads but does not
    run must not end the search."""
    if os.name == "nt":
        pytest.skip("POSIX shell script")
    payloads = {"arm.xz": lzma.compress(b"#!/bin/sh\nexit 0\n"), "intel.xz": lzma.compress(_FAKE)}

    def fake_download(url, dest):
        name = url.split("/")[-1]
        if name.endswith(".sha256"):
            dest.write_text(hashlib.sha256(payloads[name[:-7]]).hexdigest())
        else:
            dest.write_bytes(payloads[name])
    monkeypatch.setattr(binaries, "_download", fake_download)
    monkeypatch.setattr(binaries, "_release_assets", lambda: ["arm.xz", "intel.xz"])
    monkeypatch.setenv("TARQUIN_CACHE_DIR", str(tmp_path / "root"))

    got = binaries._download_release_binary(tmp_path)
    assert got.read_bytes() == _FAKE
    assert len(list((tmp_path / "root" / "quarantine").iterdir())) == 1


def test_release_download_rejects_a_bad_checksum(monkeypatch, tmp_path):
    """A release asset is a plain URL anyone with write access could re-upload; the
    sidecar hash ties it to the CI run that built it, so a mismatch must not install."""
    _serve(monkeypatch, lzma.compress(b"\x7fELF not really"), "0" * 64)
    assert binaries._download_release_binary(tmp_path) is None
    assert not (tmp_path / binaries._exec_name()).exists()


#********************#
#   container shim   #
#********************#
def test_shim_is_recognised_and_executable(tmp_path):
    if os.name == "nt":
        pytest.skip("shim is POSIX only")
    shim = binaries._write_shim(tmp_path, "docker", "example.org/tarquin:v1")
    assert binaries.is_container_shim(shim)
    assert not binaries.is_container_shim(tmp_path / "tarquin")
    assert os.access(shim, os.X_OK)
    text = shim.read_text()
    assert "example.org/tarquin:v1" in text
    assert text.startswith("#!/bin/sh")
    assert "tarquin_wrapper.container import main" in text
    assert binaries.is_container_shim(tmp_path / "tarquin-container.cmd")   # Windows name


def test_shim_forwards_tarquins_arguments_and_mounts_cwd_at_the_same_path(tmp_path,
                                                                          monkeypatch):
    """The whole trick: TARQUIN's arguments carry absolute host paths, so they pass
    through untouched and the working directory appears under exactly the same path."""
    log = tmp_path / "args.log"
    engine = _fake_engine(tmp_path, f'printf "%s\\n" "$@" > "{log}"\n')
    home = tmp_path / "home"
    home.mkdir()
    work = tmp_path / "elsewhere" / "project"
    work.mkdir(parents=True)

    shim = binaries._write_shim(tmp_path, str(engine), "img:v1")
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("PYTHONPATH", _REPO)
    monkeypatch.delenv("TARQUIN_DOCKER_IMAGE", raising=False)
    subprocess.run([str(shim), "--input", f"{work}/in dpt.dpt", "--format", "dpt"],
                   cwd=work, check=True, stdin=subprocess.DEVNULL)

    args = log.read_text().splitlines()
    cwd = os.path.realpath(work)
    assert args[:3] == ["run", "--rm", "--init"]
    assert args[-5:] == ["img:v1", "--input", f"{work}/in dpt.dpt", "--format", "dpt"]
    assert args[args.index("-w") + 1] in (str(work), cwd)
    mounts = [args[j + 1] for j, a in enumerate(args) if a == "-v"]
    assert any(m in (f"{work}:{work}", f"{cwd}:{cwd}") for m in mounts)
    # working directory outside $HOME: the home directory is mounted as well, since
    # that is where the basis set (and the wrapper's run folder) lives
    assert f"{home}:{home}" in mounts


def test_shim_mounts_home_whole_when_the_work_lies_inside_it(tmp_path, monkeypatch):
    """Docker refuses a mount point twice, so a working directory under $HOME adds no
    mount of its own - and $HOME stays mounted whole, so a basis set or the wrapper's run
    folder elsewhere in it is still seen."""
    log = tmp_path / "args.log"
    engine = _fake_engine(tmp_path, f'printf "%s\\n" "$@" > "{log}"\n')
    home = tmp_path / "home"
    work = home / "project"
    work.mkdir(parents=True)

    shim = binaries._write_shim(tmp_path, str(engine), "img:v1")
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("PYTHONPATH", _REPO)
    subprocess.run([str(shim), "--help"], cwd=work, check=True, stdin=subprocess.DEVNULL)

    args = log.read_text().splitlines()
    mounts = [args[j + 1] for j, a in enumerate(args) if a == "-v"]
    assert mounts == [f"{home}:{home}"]
    assert args[args.index("-w") + 1] == str(work)


def test_shim_passes_the_health_probe_when_the_container_answers(tmp_path, monkeypatch):
    """The shim is verified by the very same probe as a native binary, so a container
    that prints TARQUIN's start line is accepted and one that stays silent is not."""
    monkeypatch.setenv("PYTHONPATH", _REPO)
    good_dir, silent_dir = tmp_path / "good", tmp_path / "silent"
    good_dir.mkdir()
    silent_dir.mkdir()
    good = _fake_engine(good_dir, 'echo "2026-Oct-01 INFO   : TARQUIN 4.3.11 Started"; exit 255\n')
    silent = _fake_engine(silent_dir, 'exit 0\n')

    ok, why = binaries.verify_executable(binaries._write_shim(good_dir, str(good), "img"))
    assert ok is True, why
    ok, _ = binaries.verify_executable(binaries._write_shim(silent_dir, str(silent), "img"))
    assert ok is False


def test_container_can_see_only_cwd_and_home(tmp_path, monkeypatch):
    if os.name == "nt":
        pytest.skip("Windows mounts whole drives")
    home = tmp_path / "home"
    work = tmp_path / "work"
    other = tmp_path / "other"
    for d in (home, work, other):
        d.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.chdir(work)
    assert container.can_see(work / "tmp" / "temp0.dpt")
    assert container.can_see(home / "basis" / "x.basis")
    assert container.can_see(home)
    assert not container.can_see(other / "x.basis")
    # a working directory inside $HOME does not narrow what is seen to itself
    (home / "project").mkdir()
    monkeypatch.chdir(home / "project")
    assert container.can_see(home / ".cache" / "tarquin_wrapper" / "runs" / "temp0.dpt")
    # a stale $PWD names no mount
    monkeypatch.setenv("PWD", str(other))
    assert not container.can_see(other / "x.basis")


def test_cached_shim_is_skipped_not_quarantined_when_engine_is_down(tmp_path, monkeypatch):
    if os.name == "nt":
        pytest.skip("shim is POSIX only")
    shim = binaries._write_shim(tmp_path, "docker", "img")
    monkeypatch.setattr(binaries, "_container_cli", lambda: "docker")
    monkeypatch.setattr(binaries, "_container_ready", lambda cli, timeout=30.0: (False, "down"))
    assert binaries._cached_binary(tmp_path) is None
    assert shim.is_file()                     # still there for next time
    monkeypatch.setattr(binaries, "_container_ready", lambda cli, timeout=30.0: (True, "ok"))
    assert binaries._cached_binary(tmp_path) == shim


def test_native_cache_wins_over_the_shim(tmp_path):
    if os.name == "nt":
        pytest.skip("shim is POSIX only")
    binaries._write_shim(tmp_path, "docker", "img")
    native = _make_elf(tmp_path / binaries._exec_name(), 0x3E)
    assert binaries._cached_binary(tmp_path) == native


@pytest.mark.parametrize("host, inside", [
    (r"C:\Users\me\proj\tmp\temp0.dpt", "/host/C/Users/me/proj/tmp/temp0.dpt"),
    (r"c:\Users\me\basis.BASIS",        "/host/C/Users/me/basis.BASIS"),   # drive upper-cased
    (r"D:\data\x.dpt",                  "/host/D/data/x.dpt"),
    ("C:\\",                            "/host/C/"),
])
def test_windows_paths_map_under_host(host, inside):
    assert container.windows_to_container(host) == inside


def test_windows_unc_paths_are_refused():
    with pytest.raises(ValueError, match="UNC"):
        container.windows_to_container(r"\\server\share\x.dpt")


def test_arguments_with_a_drive_letter_are_translated_and_nothing_else():
    """Only paths are rewritten - option names and values such as 'true', '0.03' or
    'slaser' pass as they are - and the input list is left untouched."""
    args = ["--input", r"C:\Users\me\proj\temp0.dpt", "--format", "dpt",
            "--basis_lcm", r"D:\bases\press.BASIS", "--echo", "0.03", "--auto_phase", "true"]
    before = list(args)
    out = container.translate_args(args, mapper=container.windows_to_container)
    assert args == before
    assert out == ["--input", "/host/C/Users/me/proj/temp0.dpt", "--format", "dpt",
                   "--basis_lcm", "/host/D/bases/press.BASIS", "--echo", "0.03",
                   "--auto_phase", "true"]


def test_only_drive_paths_are_translated():
    """A value with a colon is no path unless a separator follows the drive letter."""
    args = ["--title", "T:2 weighted", "--input", "C:/data/x.dpt", "--x", "C:rel\\x.dpt"]
    out = container.translate_args(args, mapper=container.windows_to_container)
    assert out == ["--title", "T:2 weighted", "--input", "/host/C/data/x.dpt", "--x",
                   "C:rel\\x.dpt"]


def test_the_container_runs_with_an_init():
    """TARQUIN as the container's first process ignores SIGTERM and SIGINT; with an init
    in front, Ctrl-C and the wrapper's timeout stop it."""
    assert "--init" in container.run_command("docker", "img", [], cwd="/data", home="/home/me",
                                             windows=False)


def test_mounts_never_include_the_root():
    """Docker refuses '/' as a mount point (a home of '/' under some service accounts)."""
    if os.name == "nt":
        pytest.skip("POSIX mounts")
    assert container.posix_mounts("/", "/") == []
    assert container.posix_mounts("/data/run", "/") == [("/data/run", "/data/run")]


def test_a_symlinked_home_is_mounted_under_both_spellings(tmp_path):
    real = tmp_path / "gpfs" / "me"
    real.mkdir(parents=True)
    (tmp_path / "home").mkdir()
    link = tmp_path / "home" / "me"
    link.symlink_to(real)
    mounts = [host for host, _ in container.posix_mounts(str(link / "run"), str(link))]
    assert str(link) in mounts and str(real) in mounts


def test_translation_is_identity_on_posix():
    if os.name == "nt":
        pytest.skip("identity mounts are the POSIX design")
    args = ["--input", "/home/me/tmp/temp0.dpt", "--basis_lcm", "/home/me/press.basis"]
    assert container.translate_args(args) == args


def test_run_command_windows_mounts_drives_under_host():
    """The Windows launcher cannot run here, but the command it would issue can be
    checked: drives of the cwd and home mounted at /host/<LETTER>, paths translated."""
    cmd = container.run_command("docker", "img:v1", ["--input", r"D:\proj\run\temp0.dpt"],
                                cwd=r"D:\proj\run", home=r"C:\Users\me", windows=True)
    assert cmd[:4] == ["docker", "run", "--rm", "--init"]
    assert "-u" not in cmd                                   # no uid mapping on Windows
    mounts = [cmd[i + 1] for i, a in enumerate(cmd) if a == "-v"]
    assert mounts == ["D:\\:/host/D", "C:\\:/host/C"]
    assert cmd[cmd.index("-w") + 1] == "/host/D/proj/run"
    assert cmd[-3:] == ["img:v1", "--input", "/host/D/proj/run/temp0.dpt"]


def test_run_command_posix_uses_identical_paths(tmp_path):
    home, data = tmp_path / "home", tmp_path / "data" / "run"   # real folders: realpath is asked
    (home / "run").mkdir(parents=True)
    data.mkdir(parents=True)
    cmd = container.run_command("docker", "img:v1", ["--help"], cwd=str(data),
                                home=str(home), windows=False, uid=(1000, 1000))
    assert cmd[cmd.index("-u") + 1] == "1000:1000"
    mounts = [cmd[i + 1] for i, a in enumerate(cmd) if a == "-v"]
    assert mounts == [f"{home}:{home}", f"{data}:{data}"]
    assert cmd[cmd.index("-w") + 1] == str(data)
    assert cmd[-2:] == ["img:v1", "--help"]
    # podman: uid mapping via user namespace instead of -u
    cmd = container.run_command("podman", "img", [], cwd=str(home / "run"), home=str(home),
                                windows=False, uid=(1000, 1000))
    assert "--userns=keep-id" in cmd and "-u" not in cmd
    mounts = [cmd[i + 1] for i, a in enumerate(cmd) if a == "-v"]
    assert mounts == [f"{home}:{home}"]                          # cwd inside home: no 2nd mount


def test_container_rung_is_opt_out(monkeypatch, tmp_path):
    monkeypatch.setenv("TARQUIN_NO_DOCKER", "1")
    assert binaries._container_shim(tmp_path) is None
    monkeypatch.delenv("TARQUIN_NO_DOCKER")
    monkeypatch.setattr(binaries, "_container_cli", lambda: None)
    assert binaries._container_shim(tmp_path) is None


#******************#
#   cache layout   #
#******************#
def test_cache_dir_is_release_and_architecture_keyed(monkeypatch, tmp_path):
    """After an upgrade, or with TARQUIN_RELEASE_TAG, the binary of that release is used,
    not one cached for another."""
    monkeypatch.setenv("TARQUIN_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("TARQUIN_RELEASE_TAG", "v9.9.9")
    cache = binaries._cache_dir()
    assert cache.parent == tmp_path / "v9.9.9"
    assert platform.system().lower() in cache.name
    assert binaries._normalized_machine() in cache.name


def test_quarantine_moves_rather_than_deletes(monkeypatch, tmp_path):
    monkeypatch.setenv("TARQUIN_CACHE_DIR", str(tmp_path))
    bad = _make_elf(tmp_path / "tarquin", 0x3E)
    dest = binaries._quarantine(bad, reason="release")
    assert dest is not None and dest.is_file()
    assert not bad.exists()


#**************#
#   resolver   #
#**************#
def test_resolve_reports_every_source_it_tried(monkeypatch, tmp_path):
    monkeypatch.setenv("TARQUIN_CACHE_DIR", str(tmp_path))
    monkeypatch.delenv("TARQUIN_EXEC", raising=False)   # an explicit binary would be used as is
    with pytest.raises(RuntimeError) as excinfo:
        binaries.resolve_executable(allow_download=False, allow_docker=False)
    message = str(excinfo.value)
    assert "tried:" in message
    assert "cached" in message
    assert "docker=False" in message


def test_resolve_rejects_an_unrunnable_explicit_path(tmp_path):
    bad = _make_elf(tmp_path / "mytarquin", 0x3E)
    with pytest.raises(RuntimeError, match="not a runnable TARQUIN binary"):
        binaries.resolve_executable(path2exec=str(bad))


def test_resolve_still_rejects_a_missing_explicit_path(tmp_path, monkeypatch):
    with pytest.raises(FileNotFoundError, match="path2exec does not exist"):
        binaries.resolve_executable(path2exec=str(tmp_path / "nope"))
    monkeypatch.setenv("TARQUIN_EXEC", str(tmp_path / "nope"))   # named as it was given
    with pytest.raises(FileNotFoundError, match="TARQUIN_EXEC does not exist"):
        binaries.resolve_executable()


def test_a_tarquin_on_path_is_used_and_never_quarantined(tmp_path, monkeypatch):
    """A distribution's TARQUIN (snap, AUR) is found on PATH; one that does not run is
    passed over but left where it is, since it is not ours to move."""
    if os.name == "nt":
        pytest.skip("POSIX shell script")
    monkeypatch.delenv("TARQUIN_EXEC", raising=False)
    monkeypatch.setenv("TARQUIN_CACHE_DIR", str(tmp_path / "cache"))
    good = tmp_path / "good"
    good.mkdir()
    (good / "tarquin").write_bytes(_FAKE)
    os.chmod(good / "tarquin", 0o755)
    monkeypatch.setenv("PATH", str(good))
    assert binaries.resolve_executable(allow_download=False, allow_docker=False) == \
        str((good / "tarquin").resolve())

    bad = tmp_path / "bad"
    bad.mkdir()
    (bad / "tarquin").write_text("#!/bin/sh\nexit 0\n")
    os.chmod(bad / "tarquin", 0o755)
    monkeypatch.setenv("PATH", str(bad))
    with pytest.raises(RuntimeError, match="PATH: ran"):
        binaries.resolve_executable(allow_download=False, allow_docker=False)
    assert (bad / "tarquin").is_file()


def test_no_container_means_no_cached_shim_either(tmp_path, monkeypatch):
    if os.name == "nt":
        pytest.skip("shim is POSIX only")
    binaries._write_shim(tmp_path, "docker", "img")
    monkeypatch.setattr(binaries, "_container_cli", lambda: "docker")
    monkeypatch.setattr(binaries, "_container_ready", lambda cli, timeout=30.0: (True, "ok"))
    assert binaries._cached_binary(tmp_path, allow_docker=False) is None
    assert binaries.is_container_shim(binaries._cached_binary(tmp_path))


#*************#
#   version   #
#*************#
def test_the_version_is_read_from_the_start_line(tmp_path):
    if os.name == "nt":
        pytest.skip("POSIX shell script")
    fake = tmp_path / "tarquin"
    fake.write_bytes(_FAKE)
    os.chmod(fake, 0o755)
    assert binaries.version(fake) == "4.3.11"
    assert binaries.version(sys.executable) is None
    assert binaries.version(tmp_path / "missing") is None


@pytest.mark.skipif(_real_binary() is None, reason="no TARQUIN binary")
def test_the_real_binary_names_its_release():
    assert binaries.version(_real_binary()) == "4.3.11"
