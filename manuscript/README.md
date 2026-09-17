# manuscript/ (git-ignored)

Drop the current manuscript `.docx` here to let

    python analysis/verify_manuscript_numbers.py

run its wording checks (it picks the newest `.docx` in this folder), or pass
an explicit path with `--manuscript PATH`. Nothing outside the repository is
read: with no file here the wording checks are skipped with a message and only
the output checks run. Everything except this README is ignored by git.
