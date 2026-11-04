"""Host CLI Orchestrator for Headless Unreal Engine Data Generation.

Locates Unreal Engine 5 binaries, builds the command-line arguments,
launches UnrealEditor-Cmd in headless mode, and streams capture logs.
"""

import os
import sys
import subprocess
import glob
from typing import Optional, List


def find_unreal_editor_executable() -> Optional[str]:
    """Auto-detects UnrealEditor-Cmd.exe on Windows / Linux systems."""
    # Standard installation paths for Unreal Engine 5 on Windows
    patterns = [
        "C:/Program Files/Epic Games/UE_5.*/Engine/Binaries/Win64/UnrealEditor-Cmd.exe",
        "D:/Epic Games/UE_5.*/Engine/Binaries/Win64/UnrealEditor-Cmd.exe",
        "C:/Epic Games/UE_5.*/Engine/Binaries/Win64/UnrealEditor-Cmd.exe",
        "C:/Program Files/Epic Games/UE_4.*/Engine/Binaries/Win64/UE4Editor-Cmd.exe",
    ]

    for pat in patterns:
        matches = glob.glob(pat)
        if matches:
            # Sort to pick highest version
            matches.sort(reverse=True)
            return matches[0]

    return None


class UnrealHeadlessRunner:
    """Manages spawning headless Unreal Engine instances to execute Python capture scripts."""

    def __init__(
        self,
        editor_exe: Optional[str] = None,
        project_uproject: Optional[str] = None
    ):
        self.editor_exe = editor_exe or find_unreal_editor_executable()
        self.project_uproject = project_uproject

    def build_command(
        self,
        python_script_path: str,
        script_args: Optional[List[str]] = None,
        null_rhi: bool = False
    ) -> List[str]:
        """Constructs UnrealEditor-Cmd execution command."""
        if not self.editor_exe:
            raise FileNotFoundError("UnrealEditor executable not found. Please install Unreal Engine 5.")

        cmd = [
            self.editor_exe,
            self.project_uproject or "",
            "-run=pythonscript",
            f"-script={python_script_path}",
            "-unattended",
            "-nopause",
            "-nosplash",
            "-stdout"
        ]

        if null_rhi:
            cmd.append("-nullrhi")

        if script_args:
            cmd.extend(script_args)

        return cmd

    def run_script(
        self,
        python_script_path: str,
        script_args: Optional[List[str]] = None,
        null_rhi: bool = False
    ) -> int:
        """Executes the Python script inside Unreal Engine and prints stdout in real time."""
        abs_script = os.path.abspath(python_script_path)
        cmd = self.build_command(abs_script, script_args, null_rhi=null_rhi)

        print(f"[UE_RUNNER] Executing: {' '.join(cmd)}")
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )

        for line in proc.stdout:
            print(f"[UE] {line.strip()}")

        proc.wait()
        return proc.returncode

