import subprocess

cmd = [
    "iperf3",
    "-u",
    "-c", "10.45.0.1",
    "-B", "10.45.0.2",
    "-p", "5201",
    "-b", "20M",
    "-t", "10",
    "-i", "1"
]

subprocess.run(cmd)
