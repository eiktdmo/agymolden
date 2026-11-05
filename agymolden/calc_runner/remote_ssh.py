"""
Remote SSH & VPN Connection Manager and Runner for Quantum Chemistry Calculations.

Features:
- Reads ~/.ssh/config hosts and parses bash history / known connections dynamically (learning past connections)
- Tests reachability of remote host via SSH over VPN
- Transfers input files via scp/rsync
- Executes remote calculations (direct python wrapper or sbatch SLURM)
- Streams stdout live back into JobMonitorDialog for real-time 3D viewport updates
- Syncs generated artifacts (.out, _final.xyz, .molden, .cube) back to local directory
"""

import os
import re
import shlex
import subprocess
import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any

from PyQt6 import QtCore, QtWidgets

from .base import BaseCalcRunner, JobStatus, OptimizationStep
from agymolden.elements import ELEMENT_DATA

log = logging.getLogger(__name__)


@dataclass
class RemoteHostConfig:
    alias: str = ""
    hostname: str = ""
    user: str = ""
    identity_file: str = ""
    password: str = ""
    port: int = 22
    remote_workdir: str = ""
    remote_python: str = "python3"
    remote_wrapper: str = "newBRORSEN.v04.09.py"
    vpn_command: str = ""
    use_nice: bool = True
    nice_value: int = 10
    use_nohup: bool = True
    use_slurm: bool = False
    slurm_partition: str = ""
    slurm_cpus: int = 4
    slurm_mem: str = "8GB"
    slurm_time: str = "24:00:00"


def parse_ssh_config_hosts() -> List[RemoteHostConfig]:
    """Parse ~/.ssh/config to discover configured hosts."""
    hosts = []
    cfg_path = os.path.expanduser("~/.ssh/config")
    if not os.path.isfile(cfg_path):
        return hosts

    current_host: Optional[RemoteHostConfig] = None

    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split(None, 1)
                if len(parts) < 2:
                    continue
                key, val = parts[0].lower(), parts[1].strip()

                if key == "host":
                    if current_host and current_host.alias != "*":
                        hosts.append(current_host)
                    current_host = RemoteHostConfig(
                        alias=val,
                        hostname=val,
                        user=os.environ.get("USER", "andres"),
                    )
                elif current_host:
                    if key == "hostname":
                        current_host.hostname = val
                    elif key == "user":
                        current_host.user = val
                    elif key == "identityfile":
                        current_host.identity_file = os.path.expanduser(val)
                    elif key == "port":
                        try:
                            current_host.port = int(val)
                        except ValueError:
                            pass

        if current_host and current_host.alias != "*":
            hosts.append(current_host)
    except Exception as e:
        log.warning(f"Error parsing ~/.ssh/config: {e}")

    return hosts


def parse_history_remote_connections() -> List[RemoteHostConfig]:
    """Inspect bash history for recent user@host and ssh key connections (e.g. echamorro@192.168.2.106)."""
    learned = []
    hist_path = os.path.expanduser("~/.bash_history")
    if not os.path.isfile(hist_path):
        return learned

    seen = set()
    try:
        with open(hist_path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()

        # Iterate in reverse to capture most recent entries first
        for line in reversed(lines):
            line = line.strip()
            if not any(line.startswith(cmd) for cmd in ("ssh ", "scp ", "sftp ")):
                continue

            # Look for user@host
            m_uh = re.search(r"([a-zA-Z0-9_\.\-]+)@([a-zA-Z0-9_\.\-]+)", line)
            if not m_uh:
                continue
            u, h = m_uh.group(1), m_uh.group(2)
            key_tag = f"{u}@{h}"
            if key_tag in seen:
                continue
            seen.add(key_tag)

            # Check for -i identity_file
            m_key = re.search(r"-i\s+([^\s]+)", line)
            ident = os.path.expanduser(m_key.group(1)) if m_key else ""
            if not ident:
                # Check default identity files if they match
                cand_key = os.path.expanduser("~/.ssh/id_rsa_totalcmd.pem")
                if os.path.isfile(cand_key) and "192.168.2.106" in h:
                    ident = cand_key

            learned.append(RemoteHostConfig(
                alias=f"{u}@{h} (Learned from session history)",
                hostname=h,
                user=u,
                identity_file=ident,
                remote_workdir=f"/home/{u}/pruebareyes",
            ))
    except Exception as e:
        log.warning(f"Error reading bash history: {e}")

    return learned


def get_all_remote_hosts() -> List[RemoteHostConfig]:
    """Combine saved user profiles from QSettings, SSH config hosts, and learned history connections."""
    all_hosts: List[RemoteHostConfig] = []
    seen = set()

    # 1. Load saved user profiles from QSettings FIRST
    try:
        settings = QtCore.QSettings("AgyMolden", "RemoteProfiles")
        for group in settings.childGroups():
            if not group.startswith("profile_"):
                continue
            h_name = settings.value(f"{group}/hostname", "")
            if not h_name:
                continue
            cfg = RemoteHostConfig(
                alias=settings.value(f"{group}/alias", h_name),
                hostname=h_name,
                user=settings.value(f"{group}/user", "andres"),
                identity_file=settings.value(f"{group}/identity_file", ""),
                remote_workdir=settings.value(f"{group}/remote_workdir", "~/agymolden_scratch"),
                remote_python=settings.value(f"{group}/remote_python", "python3"),
                remote_wrapper=settings.value(f"{group}/remote_wrapper", "newBRORSEN.v04.09.py"),
                vpn_command=settings.value(f"{group}/vpn_command", ""),
                use_nice=(str(settings.value(f"{group}/use_nice", "true")).lower() == "true"),
                nice_value=int(settings.value(f"{group}/nice_value", 10)),
                use_nohup=(str(settings.value(f"{group}/use_nohup", "true")).lower() == "true"),
                use_slurm=(str(settings.value(f"{group}/use_slurm", "false")).lower() == "true"),
                slurm_partition=str(settings.value(f"{group}/slurm_partition", "")),
                slurm_cpus=int(settings.value(f"{group}/slurm_cpus", 4)),
                slurm_mem=str(settings.value(f"{group}/slurm_mem", "8GB")),
                slurm_time=str(settings.value(f"{group}/slurm_time", "24:00:00")),
            )
            key = cfg.alias or f"{cfg.user}@{cfg.hostname}"
            if key not in seen:
                all_hosts.append(cfg)
                seen.add(key)
                seen.add(cfg.hostname)
    except Exception as e:
        log.warning(f"Error reading QSettings RemoteProfiles: {e}")

    # 2. Add ~/.ssh/config hosts
    for h in parse_ssh_config_hosts():
        key = f"{h.user}@{h.hostname}"
        if key not in seen:
            all_hosts.append(h)
            seen.add(key)

    # 3. Add learned connections from bash history
    for h in parse_history_remote_connections():
        key = f"{h.user}@{h.hostname}"
        if key not in seen:
            all_hosts.append(h)
            seen.add(key)

    return all_hosts


def _wrap_ssh_command(base_cmd: List[str], password: str = "") -> Tuple[List[str], Dict[str, str]]:
    """Wrap command with sshpass when password authentication is used."""
    import shutil
    env = dict(os.environ)
    if not password:
        return base_cmd, env

    sshpass_bin = shutil.which("sshpass")
    if sshpass_bin:
        env["SSHPASS"] = password
        # Use sshpass -e (reads password from SSHPASS env var safely)
        return [sshpass_bin, "-e"] + base_cmd, env

    return base_cmd, env


def test_ssh_connectivity(host_cfg: RemoteHostConfig, timeout_sec: int = 4) -> Tuple[bool, str]:
    """Test SSH connection to remote host over active network / VPN."""
    if not host_cfg.hostname:
        return False, "Host / IP is empty. Please enter a remote hostname."

    import shutil
    cmd = [
        "ssh",
        "-o", f"ConnectTimeout={timeout_sec}",
        "-o", "ServerAliveInterval=5",
        "-o", "ServerAliveCountMax=2",
    ]
    if not host_cfg.password:
        cmd.extend(["-o", "BatchMode=yes"])
    else:
        # Avoid hanging on StrictHostKeyChecking prompts during password automation
        cmd.extend(["-o", "StrictHostKeyChecking=accept-new"])

    if host_cfg.identity_file and os.path.isfile(host_cfg.identity_file):
        cmd.extend(["-i", host_cfg.identity_file, "-o", "IdentitiesOnly=yes"])
    if host_cfg.port != 22:
        cmd.extend(["-p", str(host_cfg.port)])

    target = f"{host_cfg.user}@{host_cfg.hostname}" if host_cfg.user else host_cfg.hostname
    cmd.extend([target, "echo READY"])

    full_cmd, env = _wrap_ssh_command(cmd, host_cfg.password)
    if host_cfg.password and not shutil.which("sshpass"):
        return False, "Password entered, but 'sshpass' utility is not installed. Install it with: sudo apt install sshpass, or configure SSH key authentication."

    try:
        r = subprocess.run(full_cmd, capture_output=True, text=True, timeout=timeout_sec + 3, env=env)
        if r.returncode == 0 and "READY" in r.stdout:
            return True, f"Connected to {target} successfully."
        else:
            err = r.stderr.strip() or r.stdout.strip() or f"Exited with code {r.returncode}"
            return False, f"Could not reach {target}: {err}"
    except subprocess.TimeoutExpired:
        return False, f"Connection timed out (Host {host_cfg.hostname} unreachable. Check VPN status)."
    except Exception as e:
        return False, str(e)


def generate_slurm_script(
    job_name: str,
    input_file: str,
    remote_python: str,
    remote_script: str,
    extra_flags: List[str],
    remote_workdir: str,
    cpus: int = 4,
    mem: str = "8GB",
    walltime: str = "24:00:00",
    partition: str = "",
    nice_prefix: str = "",
) -> str:
    """Generate a standard SLURM batch job submission script (.slurm)."""
    clean_dir = remote_workdir.replace("~/", "") if remote_workdir.startswith("~/") else remote_workdir
    stem = os.path.splitext(os.path.basename(input_file))[0]
    part_line = f"#SBATCH --partition={partition}\n" if partition else ""
    return (
        "#!/bin/bash\n"
        f"#SBATCH --job-name={job_name or stem}\n"
        f"#SBATCH --output={stem}.out\n"
        f"#SBATCH --error={stem}.err\n"
        "#SBATCH --nodes=1\n"
        f"#SBATCH --cpus-per-task={cpus}\n"
        f"#SBATCH --mem={mem}\n"
        f"#SBATCH --time={walltime}\n"
        f"{part_line}"
        "\n"
        f"cd {clean_dir}\n"
        f"{nice_prefix}{remote_python} -u {remote_script} {' '.join(extra_flags)} {os.path.basename(input_file)}\n"
    )


class RemoteSshRunner(BaseCalcRunner):
    """Executes calculation remotely via SSH and streams live output back to local GUI."""

    def __init__(self, parent: Optional[QtCore.QObject] = None):
        super().__init__(parent)
        self.host_cfg: Optional[RemoteHostConfig] = None
        self.remote_script: str = "newBRORSEN.v04.09.py"
        self.extra_flags: List[str] = []
        self._curr_cycle: int = 0
        self._geom_buffer: List[Tuple[str, float, float, float]] = []
        self._parsing_geom: bool = False

    def configure_remote(self, host_cfg: RemoteHostConfig, remote_script: str = "newBRORSEN.v04.09.py", extra_flags: Optional[List[str]] = None):
        self.host_cfg = host_cfg
        self.remote_script = remote_script
        self.extra_flags = extra_flags or []

    def start_job(self, input_file: str, work_dir: Optional[str] = None, **kwargs) -> bool:
        if not self.host_cfg:
            self._set_status(JobStatus.FAILED, "No remote host configured.")
            return False

        if not os.path.isfile(input_file):
            self._set_status(JobStatus.FAILED, f"Local input file not found: {input_file}")
            return False

        self.input_file = os.path.abspath(input_file)
        self.work_dir = work_dir or os.path.dirname(self.input_file)
        self.steps.clear()
        self._raw_log_buffer.clear()
        self._curr_cycle = 0
        self._geom_buffer.clear()
        self._parsing_geom = False

        base_in = os.path.basename(self.input_file)
        rem_dir = self.host_cfg.remote_workdir
        target = f"{self.host_cfg.user}@{self.host_cfg.hostname}" if self.host_cfg.user else self.host_cfg.hostname

        self._set_status(JobStatus.STARTING, f"Connecting to {self.host_cfg.hostname} & transferring files...")

        import threading

        def _remote_prep_worker():
            clean_rem_dir = rem_dir.replace("~/", "") if rem_dir.startswith("~/") else rem_dir
            scp_cmd = ["scp", "-o", "ConnectTimeout=8", "-o", "ServerAliveInterval=5", "-o", "ServerAliveCountMax=2"]
            if self.host_cfg.identity_file and os.path.isfile(self.host_cfg.identity_file):
                scp_cmd.extend(["-i", self.host_cfg.identity_file, "-o", "IdentitiesOnly=yes"])
            if self.host_cfg.port != 22:
                scp_cmd.extend(["-P", str(self.host_cfg.port)])

            # Ensure remote workdir exists first (with fast 8s timeout)
            mkdir_cmd = ["ssh", "-o", "ConnectTimeout=5", "-o", "ServerAliveInterval=5", "-o", "ServerAliveCountMax=2"]
            if not self.host_cfg.password:
                mkdir_cmd.extend(["-o", "BatchMode=yes"])
            else:
                mkdir_cmd.extend(["-o", "StrictHostKeyChecking=accept-new"])
            if self.host_cfg.identity_file and os.path.isfile(self.host_cfg.identity_file):
                mkdir_cmd.extend(["-i", self.host_cfg.identity_file, "-o", "IdentitiesOnly=yes"])
            if self.host_cfg.port != 22:
                mkdir_cmd.extend(["-p", str(self.host_cfg.port)])
            q_rem_dir = shlex.quote(clean_rem_dir)
            mkdir_cmd.extend([target, f"mkdir -p {q_rem_dir}"])
            full_mkdir, mkdir_env = _wrap_ssh_command(mkdir_cmd, self.host_cfg.password)
            try:
                subprocess.run(full_mkdir, capture_output=True, timeout=8, env=mkdir_env)
            except Exception as e:
                log.warning(f"Remote mkdir notice: {e}")

            # Copy input (options MUST come before source and destination targets!)
            if self.host_cfg.password:
                scp_cmd.extend(["-o", "StrictHostKeyChecking=accept-new"])
            else:
                scp_cmd.extend(["-o", "BatchMode=yes"])
            scp_cmd.extend([self.input_file, f"{target}:{clean_rem_dir}/{base_in}"])
            full_scp, scp_env = _wrap_ssh_command(scp_cmd, self.host_cfg.password)
            try:
                r_scp = subprocess.run(full_scp, capture_output=True, text=True, timeout=15, env=scp_env)
                if r_scp.returncode != 0:
                    return False, f"SCP upload failed: {r_scp.stderr}"
            except Exception as e:
                return False, f"SCP upload error: {e}"

            # 2. Build execution command
            ssh_bin = "ssh"
            ssh_args = [
                "-o", "ConnectTimeout=8",
                "-o", "ServerAliveInterval=10",
                "-o", "ServerAliveCountMax=3",
            ]
            if not self.host_cfg.password:
                ssh_args.extend(["-o", "BatchMode=yes"])
            else:
                ssh_args.extend(["-o", "StrictHostKeyChecking=accept-new"])
            if self.host_cfg.identity_file and os.path.isfile(self.host_cfg.identity_file):
                ssh_args.extend(["-i", self.host_cfg.identity_file, "-o", "IdentitiesOnly=yes"])
            if self.host_cfg.port != 22:
                ssh_args.extend(["-p", str(self.host_cfg.port)])

            nice_prefix = f"nice -n {int(self.host_cfg.nice_value)} " if self.host_cfg.use_nice else ""
            stem = os.path.splitext(base_in)[0]
            remote_log = f"{stem}.out"
            q_remote_log = shlex.quote(remote_log)
            q_base_in = shlex.quote(base_in)

            if self.host_cfg.use_slurm:
                slurm_file = f"{stem}.slurm"
                q_slurm_file = shlex.quote(slurm_file)
                local_slurm_path = os.path.join(self.work_dir, slurm_file)
                slurm_content = generate_slurm_script(
                    job_name=stem,
                    input_file=self.input_file,
                    remote_python=self.host_cfg.remote_python,
                    remote_script=self.remote_script,
                    extra_flags=self.extra_flags,
                    remote_workdir=clean_rem_dir,
                    cpus=self.host_cfg.slurm_cpus,
                    mem=self.host_cfg.slurm_mem,
                    walltime=self.host_cfg.slurm_time,
                    partition=self.host_cfg.slurm_partition,
                    nice_prefix=nice_prefix,
                )
                try:
                    with open(local_slurm_path, "w", encoding="utf-8") as f_slurm:
                        f_slurm.write(slurm_content)
                except Exception as e:
                    return False, f"Failed to create local SLURM script: {e}"

                scp_slurm_cmd = ["scp", "-o", "ConnectTimeout=8", "-o", "ServerAliveInterval=5", "-o", "ServerAliveCountMax=2"]
                if self.host_cfg.identity_file and os.path.isfile(self.host_cfg.identity_file):
                    scp_slurm_cmd.extend(["-i", self.host_cfg.identity_file, "-o", "IdentitiesOnly=yes"])
                if self.host_cfg.port != 22:
                    scp_slurm_cmd.extend(["-P", str(self.host_cfg.port)])
                if self.host_cfg.password:
                    scp_slurm_cmd.extend(["-o", "StrictHostKeyChecking=accept-new"])
                else:
                    scp_slurm_cmd.extend(["-o", "BatchMode=yes"])
                scp_slurm_cmd.extend([local_slurm_path, f"{target}:{clean_rem_dir}/{slurm_file}"])
                full_scp_slurm, scp_slurm_env = _wrap_ssh_command(scp_slurm_cmd, self.host_cfg.password)
                try:
                    r_slurm_scp = subprocess.run(full_scp_slurm, capture_output=True, text=True, timeout=15, env=scp_slurm_env)
                    if r_slurm_scp.returncode != 0:
                        return False, f"SLURM script SCP upload failed: {r_slurm_scp.stderr}"
                except Exception as e:
                    return False, f"SLURM script upload error: {e}"

                sbatch_cmd = ["ssh", "-o", "ConnectTimeout=8", "-o", "ServerAliveInterval=5", "-o", "ServerAliveCountMax=2"]
                if not self.host_cfg.password:
                    sbatch_cmd.extend(["-o", "BatchMode=yes"])
                else:
                    sbatch_cmd.extend(["-o", "StrictHostKeyChecking=accept-new"])
                if self.host_cfg.identity_file and os.path.isfile(self.host_cfg.identity_file):
                    sbatch_cmd.extend(["-i", self.host_cfg.identity_file, "-o", "IdentitiesOnly=yes"])
                if self.host_cfg.port != 22:
                    sbatch_cmd.extend(["-p", str(self.host_cfg.port)])
                sbatch_cmd.extend([target, f"cd {q_rem_dir} && sbatch {q_slurm_file}"])
                full_sbatch, sbatch_env = _wrap_ssh_command(sbatch_cmd, self.host_cfg.password)
                try:
                    r_sub = subprocess.run(full_sbatch, capture_output=True, text=True, timeout=20, env=sbatch_env)
                    if r_sub.returncode != 0:
                        return False, f"SLURM sbatch submission failed: {r_sub.stderr.strip() or r_sub.stdout.strip()}"
                    sub_out = r_sub.stdout.strip()
                    m_jid = re.search(r"Submitted\s+batch\s+job\s+(\d+)", sub_out, re.IGNORECASE)
                    slurm_job_id = m_jid.group(1) if m_jid else ""
                except Exception as e:
                    return False, f"SLURM submission error: {e}"

                if slurm_job_id:
                    remote_cmd = (
                        f"cd {q_rem_dir} && touch {q_remote_log} && "
                        f"(tail -n +1 -f {q_remote_log} & T_PID=$!; "
                        f"while squeue -j {slurm_job_id} 2>/dev/null | grep -q {slurm_job_id}; do sleep 2; done; "
                        f"sleep 2; kill $T_PID 2>/dev/null)"
                    )
                else:
                    remote_cmd = f"cd {q_rem_dir} && touch {q_remote_log} && tail -n +1 -f {q_remote_log}"
            else:
                flags_str = " ".join(shlex.quote(f) for f in self.extra_flags)
                if self.host_cfg.use_nohup:
                    exec_body = f"nohup {nice_prefix}{self.host_cfg.remote_python} -u {self.remote_script} {flags_str} {q_base_in} 2>&1 | tee {q_remote_log}"
                else:
                    exec_body = f"{nice_prefix}{self.host_cfg.remote_python} -u {self.remote_script} {flags_str} {q_base_in}"
                remote_cmd = f"cd {q_rem_dir} && {exec_body}"

            ssh_args.extend([target, remote_cmd])
            full_exec_cmd, exec_env = _wrap_ssh_command([ssh_bin] + ssh_args, self.host_cfg.password)
            return True, (full_exec_cmd, exec_env)

        def _on_prep_finished(res):
            ok, payload = res
            if not ok:
                self._set_status(JobStatus.FAILED, payload)
                self.finished.emit(False, payload)
                return

            full_exec_cmd, exec_env = payload
            final_bin = full_exec_cmd[0]
            final_args = full_exec_cmd[1:]
            extra_env = {"SSHPASS": self.host_cfg.password} if self.host_cfg.password else None

            self._setup_process(final_bin, final_args, self.work_dir, extra_env=extra_env)
            self._set_status(JobStatus.RUNNING, f"Running remotely on {self.host_cfg.hostname} ({base_in})...")

        # Execute prep asynchronously on a worker thread so the GUI event loop never blocks
        from ..workers import run_in_background
        run_in_background(
            getattr(self, "parent", lambda: None)() or None,
            "Remote SSH Prep",
            f"Transferring files to {self.host_cfg.hostname}...",
            _remote_prep_worker,
            _on_prep_finished,
            lambda exc: (self._set_status(JobStatus.FAILED, str(exc)), self.finished.emit(False, str(exc))),
        )
        return True

    def _on_process_finished(self, exit_code: int, exit_status):
        """When remote process terminates, pull back artifacts synchronously or defer finished signal until downloaded."""
        if self.host_cfg:
            self._pull_remote_artifacts(exit_code=exit_code, exit_status=exit_status)
        else:
            super()._on_process_finished(exit_code, exit_status)

    def _pull_remote_artifacts(self, exit_code: int = 0, exit_status: QtCore.QProcess.ExitStatus = QtCore.QProcess.ExitStatus.NormalExit):
        """Sync remote calculation outputs back to local working directory asynchronously."""
        if not self.host_cfg or not self.input_file:
            return
        target = f"{self.host_cfg.user}@{self.host_cfg.hostname}" if self.host_cfg.user else self.host_cfg.hostname
        base_no_ext = os.path.splitext(os.path.basename(self.input_file))[0]
        rem_dir = self.host_cfg.remote_workdir or "~/agymolden_scratch"
        clean_rem_dir = rem_dir.replace("~/", "") if rem_dir.startswith("~/") else rem_dir

        # Determine all relevant job stem prefixes: the input file stem AND any %jobname defined in the input file
        stem_prefixes = {base_no_ext}
        if os.path.isfile(self.input_file):
            try:
                with open(self.input_file, "r", encoding="utf-8", errors="ignore") as f_inp:
                    inp_txt = f_inp.read()
                for m_jn in re.finditer(r"^\s*%jobname\s+([^\s]+)", inp_txt, re.MULTILINE | re.IGNORECASE):
                    stem_prefixes.add(m_jn.group(1).strip())
            except Exception as e:
                log.debug("Failed parsing %%jobname stems from %s: %s", self.input_file, e)

        base_ssh_opts = ["-o", "ConnectTimeout=8", "-o", "ServerAliveInterval=5", "-o", "ServerAliveCountMax=2"]
        if self.host_cfg.password:
            base_ssh_opts.extend(["-o", "StrictHostKeyChecking=accept-new"])
        else:
            base_ssh_opts.extend(["-o", "BatchMode=yes"])
        if self.host_cfg.identity_file and os.path.isfile(self.host_cfg.identity_file):
            base_ssh_opts.extend(["-i", self.host_cfg.identity_file, "-o", "IdentitiesOnly=yes"])
        if getattr(self.host_cfg, "port", 22) != 22:
            base_ssh_opts.extend(["-p", str(self.host_cfg.port)])

        import threading

        def _pull_worker():
            out_candidate = None
            try:
                files_to_pull = []
                # 1. Query remote directory via SSH for each stem prefix
                ls_patterns = " ".join([f"{shlex.quote(st)}*" for st in stem_prefixes])
                q_rem_dir = shlex.quote(clean_rem_dir)
                list_cmd = ["ssh"] + base_ssh_opts + [target, f"cd {q_rem_dir} && ls -1 {ls_patterns}"]
                full_list_cmd, list_env = _wrap_ssh_command(list_cmd, self.host_cfg.password)
                r_ls = subprocess.run(full_list_cmd, capture_output=True, text=True, timeout=20, env=list_env)

                if r_ls.returncode == 0:
                    for line in r_ls.stdout.splitlines():
                        fname = line.strip()
                        if fname and not fname.startswith("ls:") and "cannot access" not in fname and fname not in files_to_pull:
                            files_to_pull.append(fname)

                # Fallback list of expected artifact extensions if remote ls didn't find anything
                if not files_to_pull:
                    for st in stem_prefixes:
                        for ext in (
                            ".out", ".log", ".molden", ".molden.input",
                            "_electrons.molden", "_hf_electrons.molden",
                            "_positrons.molden", "_hf_positrons.molden",
                            "_positron.molden", "_hf_positron.molden",
                            "_final.xyz", "_opt.xyz", ".xyz",
                            "_freq.txt", "_freq.molden",
                        ):
                            cand_f = f"{st}{ext}"
                            if cand_f not in files_to_pull:
                                files_to_pull.append(cand_f)

                # 2. Download files using SCP
                for fname in files_to_pull:
                    scp_opts = ["scp", "-o", "ConnectTimeout=8", "-o", "ServerAliveInterval=5", "-o", "ServerAliveCountMax=2"]
                    if self.host_cfg.password:
                        scp_opts.extend(["-o", "StrictHostKeyChecking=accept-new"])
                    else:
                        scp_opts.extend(["-o", "BatchMode=yes"])
                    if self.host_cfg.identity_file and os.path.isfile(self.host_cfg.identity_file):
                        scp_opts.extend(["-i", self.host_cfg.identity_file, "-o", "IdentitiesOnly=yes"])
                    if getattr(self.host_cfg, "port", 22) != 22:
                        scp_opts.extend(["-P", str(self.host_cfg.port)])

                    rem_file_path = f"{target}:{clean_rem_dir}/{fname}"
                    scp_opts.extend([rem_file_path, self.work_dir])
                    full_scp, scp_env = _wrap_ssh_command(scp_opts, self.host_cfg.password)
                    subprocess.run(full_scp, capture_output=True, timeout=30, env=scp_env)

                # 3. Locate downloaded .out or .log file if present
                for ext in (".out", ".log"):
                    cand = os.path.join(self.work_dir, f"{base_no_ext}{ext}")
                    if os.path.isfile(cand):
                        out_candidate = cand
                        self.log_file = cand
                        break
                self.artifacts_downloaded.emit(out_candidate or self.work_dir)
            except Exception as e:
                log.warning(f"Error pulling remote artifacts: {e}")
            finally:
                # Deliver the final job finished notification now that all calculation artifacts and Molden files are safely downloaded!
                success = (exit_code == 0 and exit_status == QtCore.QProcess.ExitStatus.NormalExit)
                if success:
                    self._set_status(JobStatus.FINISHED, f"Job finished & artifacts downloaded (code {exit_code})")
                else:
                    self._set_status(JobStatus.FAILED, f"Job terminated with exit code {exit_code}")
                self.finished.emit(success, f"Exit code {exit_code}")

        t = threading.Thread(target=_pull_worker, daemon=True)
        t.start()

    def _parse_line(self, line: str):
        """Parse streaming lines for cycles, energies, and geometries."""
        s = line.strip()

        # Check for optimization cycle header
        cycle_match = re.search(r"(?:Cycle|Step|Iteration)\s+(\d+)", s, re.IGNORECASE)
        if cycle_match:
            try:
                self._curr_cycle = int(cycle_match.group(1))
            except ValueError:
                pass

        # Energy detection
        e_match = re.search(r"(?:Total\s+Energy|E\(.+?\)|Energy)\s*[:=]\s*([+-]?\d+\.\d+)", s, re.IGNORECASE)
        if e_match:
            try:
                self._curr_energy = float(e_match.group(1))
            except ValueError:
                pass

        # Geometry block detection
        if "Coordinates (Angstrom)" in s or "Cartesian coordinates:" in s or "Standard orientation:" in s:
            self._parsing_geom = True
            self._geom_buffer.clear()
            return

        if self._parsing_geom:
            # Check for termination of geom block
            if not s or s.startswith("---") or "Grad" in s or "Energy" in s or "Cycle" in s:
                if len(self._geom_buffer) > 0:
                    self._emit_geom_step()
                self._parsing_geom = False
                return

            parts = s.split()
            if len(parts) >= 4:
                if parts[0].isdigit() and len(parts) >= 5:
                    sym, x, y, z = parts[1], parts[2], parts[3], parts[4]
                else:
                    sym, x, y, z = parts[0], parts[1], parts[2], parts[3]

                sym_clean = sym.strip().capitalize()
                if sym_clean in ELEMENT_DATA or sym.upper() in ("EP", "POS", "MU"):
                    try:
                        fx, fy, fz = float(x), float(y), float(z)
                        self._geom_buffer.append((sym, fx, fy, fz))
                    except ValueError:
                        pass

        # Convergence detection
        if "Optimization converged" in s or "CONVERGENCE REACHED" in s or "Optimization completed successfully" in s:
            self._set_status(JobStatus.CONVERGED, "Optimization converged!")
            if self.steps:
                self.steps[-1].is_converged = True
                self.step_updated.emit(self.steps[-1])

    def _emit_geom_step(self):
        if not self._geom_buffer:
            return
        syms = [item[0] for item in self._geom_buffer]
        coords = np.array([[item[1], item[2], item[3]] for item in self._geom_buffer], dtype=float)

        step = OptimizationStep(
            cycle=self._curr_cycle,
            energy=getattr(self, "_curr_energy", None),
            symbols=syms,
            coords=coords,
        )
        self.steps.append(step)
        self.step_updated.emit(step)
