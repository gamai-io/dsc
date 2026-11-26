"""Structural pre-LLM gate — the single copy in the GAMA stack.

Moved 2026-11-26 out of ``gama-fastpath/gate.py`` (and the ``T1``/``T2`` lists
out of ``gama-fastpath/fastpath.py``) so the HTTP API, the CLI and the other
pre-LLM layers stop carrying private copies.  Pure stdlib: ``os``, ``re``,
``shlex``.

The old gate was a keyword blacklist: it matched a handful of literal strings,
so flag reordering (``rm -fr /``), a shell wrapper (``sh -c '...'``), a synonym
or one shell construction was enough to pass.  This module parses the request
instead:

* shell requests are split into statements and pipelines, tokenised with
  ``shlex``, wrapper programs (``sudo``, ``env``, ``xargs``, ``timeout``, ...)
  are unwrapped, ``sh -c`` payloads and ``$(...)``/backtick substitutions are
  analysed recursively;
* SQL requests are stripped of comments and checked for statements that lose
  their scope (``delete``/``update`` without ``WHERE``, ``drop``, ``truncate``);
* disclosure requests are recognised by the *file they read* and the verb,
  not by a fixed phrase list;
* remote-code-execution and mass-mail are recognised by pipeline/argument
  structure.

Three entry points, in increasing scope:

* :func:`check_structural` — parse shell/SQL/structure only;
* :func:`check_policy` — the cheap ``T1``/``T2`` phrase pre-filter (prompt
  injection, natural-language destructive intent) that the old ``fastpath.py``
  applied after the structural gate;
* :func:`check` — the full decision: structural first, then policy.  This is
  what callers should use; it reproduces ``fastpath.optimize``'s old order
  exactly.

Every denial carries a stable ``code`` and a human ``reason``.  The regex lists
here are only a cheap pre-filter for natural-language policy phrases; they are
no longer the gate.
"""
import os
import re
import shlex

__all__ = [
    "Denied", "check", "check_structural", "check_policy", "T1", "T2",
]


class Denied:
    """A structural gate denial."""

    __slots__ = ("code", "reason")

    def __init__(self, code, reason):
        self.code = code
        self.reason = reason

    def __repr__(self):  # pragma: no cover - debugging aid
        return "Denied(%r, %r)" % (self.code, self.reason)


# ---------------------------------------------------------------------------
# vocabularies
# ---------------------------------------------------------------------------
SHELLS = {
    "sh", "bash", "zsh", "dash", "ksh", "fish", "ash", "csh", "tcsh",
    "busybox-sh", "mksh", "yash", "pwsh", "powershell",
}
INTERPRETERS = {
    "python", "python2", "python3", "perl", "ruby", "node", "nodejs", "php",
    "lua", "tclsh", "iex", "pwsh", "powershell", "Rscript",
}
DOWNLOADERS = {"curl", "wget", "fetch", "http", "https", "aria2c"}

WRAPPERS = {
    "sudo", "doas", "env", "nice", "ionice", "nohup", "time", "exec",
    "command", "builtin", "stdbuf", "timeout", "xargs", "busybox", "setsid",
    "chroot", "runuser", "su", "unbuffer", "script",
}
WRAPPER_VALUE_FLAGS = {
    "sudo": {"-u", "-g", "-p", "-C", "-h", "-U", "-r", "-t", "-T", "-D",
             "--user", "--group", "--prompt", "--close-from", "--host",
             "--other-user", "--role", "--type", "--command-timeout",
             "--chdir"},
    "doas": {"-u", "-C"},
    "su": {"-s", "-c", "-g", "-G", "--shell", "--command"},
    "runuser": {"-u", "-g", "-G", "--user", "--group", "--supp-group"},
    "timeout": {"-s", "-k", "--signal", "--kill-after"},
    "nice": {"-n", "--adjustment"},
    "ionice": {"-c", "-n", "-p", "-t", "--class", "--classdata", "--pid"},
    "stdbuf": {"-i", "-o", "-e", "--input", "--output", "--error"},
    "env": {"-u", "--unset", "-C", "--chdir", "-S", "--split-string"},
    "chroot": {"--userspec", "--groups"},
    "script": {"-c", "-t", "-I", "-O", "-B", "-E", "-T", "--command",
               "--timing", "--log-out", "--log-in", "--echo"},
}
# programs whose whole job is to move file bytes somewhere the caller can see
READ_PROGS = {
    "cat", "tac", "less", "more", "head", "tail", "grep", "egrep", "fgrep",
    "rg", "ag", "ack", "sed", "awk", "gawk", "mawk", "cut", "sort", "uniq",
    "strings", "xxd", "hexdump", "od", "base64", "cp", "mv", "scp", "rsync",
    "tar", "zip", "gzip", "bzip2", "xz", "curl", "wget", "nc", "ncat",
    "socat", "openssl", "dd", "vim", "vi", "nano", "emacs", "view",
    "python", "python2", "python3", "perl", "ruby", "node", "php",
    "type", "stat", "file", "ls", "find", "tree", "wc", "diff", "cmp",
    "jq", "yq", "source", ".",
}

# high-value secrets: any command that names one is denied
ALWAYS_SENSITIVE = re.compile(
    r"""(?ix)
      /etc/(?:shadow|gshadow|sudoers)\b
    | /etc/passwd\b
    | (?:^|/)\.ssh/id_(?:rsa|dsa|ecdsa|ed25519)\b
    | (?:^|/)id_rsa\b
    | (?:^|/)\.git-credentials\b
    | (?:^|/)\.aws/credentials\b
    | (?:^|/)\.netrc\b
    | /proc/(?:self|\d+|\*)/environ\b
    | (?:^|/)\.kube/config\b
    | (?:^|/)\.docker/config\.json\b
    """
)
# .env files: denied when read/disclosed, not when merely named in a write
ENV_SENSITIVE = re.compile(
    r"(?<![\w.-])[\w.-]*\.env(?:\.[\w-]+)?(?![\w])", re.I
)
DISCLOSE_VERB = re.compile(
    r"\b(?:prikazi|pokazi|sadrzaj|sadrzina|procitaj|ispisi|izlistaj|daj|"
    r"show|print|display|read|reveal|dump|output|contents?|content|cat)\b",
    re.I,
)
FOLD = str.maketrans("čćžšđČĆŽŠĐ", "cczsdCCZSD")


def _fold(t):
    return str(t or "").translate(FOLD)


def _norm(t):
    return re.sub(r"\s+", " ", _fold(t).lower().strip())


# ---------------------------------------------------------------------------
# policy phrases (T1 catastrophic, T2 needs explicit confirmation)
# ---------------------------------------------------------------------------
T1 = [r"\brm\s+-rf\s+[/~*]", r"\bmkfs\b", r"\bdd\s+if=.*of=/dev/", r"\bdrop\s+(table|database)\b",
      r"\btruncate\s+table\b", r"\bshutdown\b", r"\breboot\b", r"\bobrisi\s+(sve|bazu|disk)\b", r"\bizbrisi\s+bazu\b"]
T2 = [r"\bobrisi\s+(fajl|folder|log)\b", r"\bobrisi\s+/var\b", r"\bobrisi\s+.*\b(log|fajl|folder)\b",
      r"\bizbrisi\s+.*\b(log|fajl|folder)\b", r"\bdelete\s+file\b", r"\bchmod\s+777\b", r"\bchown\s+-R\b",
      r"\bkillall\b", r"\bpkill\s+-9\b",
      r"\bignorisi\s+(sva\s+)?(prethodna\s+)?(uputstv|instrukcij)", r"\bti\s+si\s+root\b", r"\bpotvrdi\s+da\s+si\s+root\b",
      r"\b(ispisi|prikazi|daj|izlistaj|podijeli|podeli)\s+(mi\s+)?(sve\s+)?(env|environment|varijabl)",
      r"\bposalji\s+(mail|email|e-mail|poruku)\b.*\b(svim|sve)\b"]
T1C = [re.compile(p, re.I) for p in T1]
T2C = [re.compile(p, re.I) for p in T2]


def check_policy(text):
    """Cheap phrase pre-filter: catastrophic (T1) then confirmation (T2)."""
    low = _norm(text)
    for p in T1C:
        if p.search(low):
            return Denied("catastrophic_command", "opasna komanda")
    for p in T2C:
        if p.search(low):
            return Denied("policy", "nije dozvoljeno bez izricite potvrde")
    return None


# ---------------------------------------------------------------------------
# shell parsing
# ---------------------------------------------------------------------------
def _split_top(s, seps, pairs=("&&", "||")):
    """Split on *seps* outside quotes and ``$(...)`` groups."""
    out, buf = [], []
    i, n = 0, len(s)
    quote = None
    depth = 0
    while i < n:
        c = s[i]
        if quote:
            buf.append(c)
            if c == "\\" and quote == '"' and i + 1 < n:
                buf.append(s[i + 1])
                i += 2
                continue
            if c == quote:
                quote = None
            i += 1
            continue
        if c in "\"'":
            quote = c
            buf.append(c)
            i += 1
            continue
        if c == "\\" and i + 1 < n:
            buf.append(c)
            buf.append(s[i + 1])
            i += 2
            continue
        if c == "$" and i + 1 < n and s[i + 1] == "(":
            depth += 1
            buf.append("$(")
            i += 2
            continue
        if depth:
            if c == "(":
                depth += 1
            elif c == ")":
                depth -= 1
            buf.append(c)
            i += 1
            continue
        pair = s[i:i + 2]
        if pair in pairs:
            if buf:
                out.append("".join(buf).strip())
                buf = []
            i += 2
            continue
        if c in seps:
            # do not split shell redirections such as 2>&1 / &>file
            if c == "&" and i > 0 and s[i - 1] in "<>":
                buf.append(c)
                i += 1
                continue
            if c == "&" and i + 1 < n and s[i + 1] == ">":
                buf.append(c)
                i += 1
                continue
            if buf:
                out.append("".join(buf).strip())
                buf = []
            i += 1
            continue
        buf.append(c)
        i += 1
    if buf:
        out.append("".join(buf).strip())
    return [x for x in out if x]


def _tokens(seg):
    try:
        return shlex.split(seg, posix=True)
    except ValueError:
        return seg.split()


def _unwrap(toks):
    """Return (program, args) after stripping assignments and wrappers."""
    i, n = 0, len(toks)
    while i < n:
        base = os.path.basename(toks[i]).lower()
        if re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", toks[i]):
            i += 1
            continue
        if base not in WRAPPERS:
            return base, toks[i + 1:]
        # consume the wrapper's own flags/operands
        start = i
        i += 1
        value_flags = WRAPPER_VALUE_FLAGS.get(base, set())
        while i < n:
            a = toks[i]
            if a == "--":
                i += 1
                break
            if a.startswith("-"):
                if a in value_flags and i + 1 < n:
                    i += 2
                else:
                    i += 1
                continue
            if re.match(r"^-?\d+(?:\.\d+)?[smhd]?$", a):
                i += 1
                continue
            if base == "env" and re.match(r"^[A-Za-z_]\w*=", a):
                i += 1
                continue
            break
        if i == start + 1 and i >= n:
            return None, []
        if i == start:  # safety: never loop forever
            i += 1
    return None, []


def _looks_absolute_root(p):
    """True for paths whose recursive deletion wipes a machine."""
    p = p.strip().strip("'\"")
    p = re.sub(r"^\$\{?HOME\}?(?=/|$)", "~", p)
    p = p.replace("//", "/")
    if p in ("/", "/*", "/.", "~", "~/", "$HOME", "${HOME}", ".", "./", "..",
             "*", ".*", "/.."):
        return True
    if p.startswith("~/") or p.startswith("$HOME/") or p.startswith("${HOME}/"):
        return True
    if re.match(r"^/(?:etc|usr|var|boot|bin|sbin|lib|lib64|opt|root|home|dev|"
                r"proc|sys|srv)(?:/|$)", p):
        return True
    return False


def _rm_denied(args):
    recursive = force = False
    targets = []
    end = False
    for a in args:
        if a == "--":
            end = True
            continue
        if not end and a.startswith("--"):
            if a == "--recursive":
                recursive = True
            elif a == "--force":
                force = True
            elif a.startswith("--no-preserve-root"):
                return Denied("destructive_root_delete",
                              "rm sa --no-preserve-root cilja korijen sistema")
            continue
        if not end and a.startswith("-") and len(a) > 1:
            letters = a[1:]
            if "r" in letters or "R" in letters:
                recursive = True
            if "f" in letters:
                force = True
            continue
        targets.append(a)
    if not recursive:
        return None
    for p in targets:
        raw = p.strip().strip("'\"")
        if _looks_absolute_root(raw):
            return Denied("destructive_root_delete",
                          "rekurzivno brisanje korijena ili kucnog direktorija: %s" % p)
        expanded = re.sub(r"^\$\{?HOME\}?(?=/|$)", "~", raw)
        if force and (expanded.startswith("/") or expanded.startswith("~")
                      or expanded in (".", "..", "*")):
            return Denied("destructive_recursive_delete",
                          "prisilno rekurzivno brisanje apsolutne putanje: %s" % p)
    return None


def _find_denied(args):
    if "-delete" in args:
        return Denied("find_delete",
                      "find -delete brise pronadjeni skup bez potvrde")
    if "-exec" in args:
        j = args.index("-exec")
        if j + 1 < len(args):
            prog = os.path.basename(args[j + 1]).lower()
            if prog in ("rm", "shred", "unlink"):
                return Denied("find_exec_delete",
                              "find -exec %s brise pronadjeni skup" % prog)
    return None


def _perm_denied(prog, args):
    recursive = any(a in ("-R", "--recursive") or
                    (a.startswith("-") and not a.startswith("--") and "R" in a[1:])
                    for a in args)
    nonflags = [a for a in args if not a.startswith("-")]
    mode = nonflags[0].lower() if nonflags else ""
    targets = nonflags[1:] if prog == "chmod" else nonflags
    for t in targets:
        if _looks_absolute_root(t):
            if recursive:
                return Denied("permissions_root",
                              "%s -R na korijenu/kucnom direktoriju: %s" % (prog, t))
            if mode in ("777", "666", "a+rwx", "o+w", "ugo+rwx"):
                return Denied("permissions_world_writable",
                              "svima upisiva dozvola na sistemskoj putanji: %s" % t)
    return None


def _sensitive_denied(prog, args, seg):
    """Deny disclosure of known secret files.

    A bare mention is not a read: the command must be a reader (``cat``,
    ``grep``, ``cp``, an interpreter, ...) or the request must use a disclosure
    verb.  This keeps "why does /etc/passwd exist" answerable.
    """
    disclosure = prog in READ_PROGS or bool(DISCLOSE_VERB.search(_fold(seg)))
    if not disclosure:
        return None
    for a in args:
        if ALWAYS_SENSITIVE.search(a):
            return Denied("sensitive_file_read",
                          "pristup osjetljivom fajlu: %s" % a)
        if ENV_SENSITIVE.search(a):
            return Denied("sensitive_file_read",
                          "citanje .env fajla: %s" % a)
    return None


def _shell_payload(args):
    for i, a in enumerate(args):
        if a == "--":
            break
        if a.startswith("-") and not a.startswith("--") and "c" in a[1:]:
            if i + 1 < len(args):
                return check_structural(args[i + 1])
        if a == "--command" and i + 1 < len(args):
            return check_structural(args[i + 1])
    return None


def _check_command(seg):
    prog, args = _unwrap(_tokens(seg))
    if not prog:
        return None
    if prog in SHELLS or prog in ("su", "runuser"):
        d = _shell_payload(args)
        if d:
            return d
    if prog == "rm":
        d = _rm_denied(args)
        if d:
            return d
    if prog == "find":
        d = _find_denied(args)
        if d:
            return d
    if prog in ("chmod", "chown", "chgrp"):
        d = _perm_denied(prog, args)
        if d:
            return d
    if prog == "dd":
        for a in args:
            if re.search(r"^of=/dev/(?:sd|hd|nvme|vd|xvd|disk|mmcblk|mapper)",
                         a, re.I):
                return Denied("disk_destroy", "dd upisuje na sirovi disk: %s" % a)
    if prog.startswith("mkfs") or prog in ("mke2fs", "wipefs", "fdisk",
                                           "parted", "sgdisk"):
        return Denied("disk_destroy", "formatiranje/particionisanje diska: %s" % prog)
    if prog == "shred" and any(a.startswith("/dev/") for a in args):
        return Denied("disk_destroy", "shred na sirovom disku")
    d = _sensitive_denied(prog, args, seg)
    if d:
        return d
    return None


def _check_remote_pipe(stages):
    progs = []
    for st in stages:
        p, _ = _unwrap(_tokens(st))
        progs.append(p)
    for i, p in enumerate(progs[:-1]):
        if p in DOWNLOADERS:
            for q in progs[i + 1:]:
                if q in SHELLS or q in INTERPRETERS:
                    return Denied(
                        "remote_code_exec",
                        "preuzeti sadrzaj se izvrsava (%s | %s)" % (p, q))
    return None


_RCE_RAW = [
    re.compile(
        r"\b(?:curl|wget|fetch)\b[^\n|;&]*\|\s*(?:sudo\s+)?(?:env\s+)?"
        r"(?:sh|bash|zsh|dash|ksh|fish|python[\d.]*|perl|ruby|node|php|"
        r"pwsh|powershell|iex)\b", re.I),
    re.compile(
        r"\b(?:sh|bash|zsh|dash|ksh)\s+(?:-\w*c\s+)?[\"']?\$\(\s*"
        r"(?:curl|wget)\b", re.I),
    re.compile(
        r"\b(?:sh|bash|zsh|dash|ksh)\s+<\(\s*(?:curl|wget)\b", re.I),
    re.compile(
        r"\b(?:eval|iex|invoke-expression)\b[^\n]*\$\(\s*(?:curl|wget)\b",
        re.I),
]

_FORK_BOMB = re.compile(
    r"(?<![\w])([A-Za-z0-9_:.-]{1,32})\s*\(\s*\)\s*\{[^{}]*"
    r"\1\s*\|\s*\1[^{}]*\}")


# ---------------------------------------------------------------------------
# SQL
# ---------------------------------------------------------------------------
def _check_sql(text):
    s = re.sub(r"--[^\n]*", " ", text)
    s = re.sub(r"/\*.*?\*/", " ", s, flags=re.S)
    s = re.sub(r"\s+", " ", s.lower())
    if re.search(r"\bdrop\s+(?:table|database|schema|view)\b", s):
        return Denied("sql_drop", "DROP uklanja cijelu tabelu/bazu")
    if re.search(r"\btruncate\s+(?:table\s+)?[\w.\"`]+", s):
        return Denied("sql_truncate", "TRUNCATE prazni cijelu tabelu")
    m = re.search(r"\bdelete\s+from\s+[^\s;]+(.*)", s)
    if m and not re.search(r"\bwhere\b", m.group(1)):
        return Denied("sql_delete_without_where",
                      "DELETE bez WHERE brise sve redove")
    m = re.search(r"\bupdate\s+\S+\s+set\b(.*)", s)
    if m and not re.search(r"\bwhere\b", m.group(1)):
        return Denied("sql_update_without_where",
                      "UPDATE bez WHERE prepisuje sve redove")
    return None


# ---------------------------------------------------------------------------
# natural language (bulk mail, destructive intent, disclosure)
# ---------------------------------------------------------------------------
_BULK_MAIL = [
    re.compile(
        r"\b(?:posalji|posaljite|posaljem|poslati|proslijedi|prosledi|"
        r"send|forward|mail|email)\b[^\n]{0,60}"
        r"\b(?:mejl|mail|email|e-mail|poruk[aeu]|poruka|obavijest|newsletter)\b"
        r"[^\n]{0,60}\b(?:svim|svima|sve|svi|svih|all|every|mass|bulk|"
        r"entire)\b", re.I),
    re.compile(
        r"\b(?:svim|svima|sve|svi|svih|all|every)\b[^\n]{0,40}"
        r"\b(?:kupc|klijent|kontakt|customer|client|contact|subscriber)\w*"
        r"[^\n]{0,40}\b(?:posalji|posaljite|poslati|send|mail|email)\b", re.I),
    re.compile(r"\bbulk\s+(?:e-?mail|mail|sms|messag)", re.I),
    re.compile(
        r"\b(?:send|posalji|posaljite)\b[^\n]{0,30}\b(?:to\s+)?\d{3,}\s+"
        r"(?:customers|clients|contacts|kupaca|klijenata)\b", re.I),
]
_DESTRUCTIVE_NL = [
    (re.compile(r"\b(?:obrisi|izbrisi|delete|remove|erase)\s+(?:sve|all)\s+"
                r"(?:fajlove|files|podatke|data|bazu|database|sadrzaj|content)",
                re.I),
     "destructive_intent", "brisanje svih podataka bez opsega"),
    (re.compile(r"\b(?:format|wipe|erase)\s+(?:the\s+)?(?:disk|drive|"
                r"filesystem|sistem)\b", re.I),
     "disk_destroy", "formatiranje diska"),
]


def _check_nl(text):
    n = _norm(text)
    for rx in _BULK_MAIL:
        if rx.search(n):
            return Denied("bulk_mail",
                          "massovno slanje maila bez izricite potvrde")
    for rx, code, reason in _DESTRUCTIVE_NL:
        if rx.search(n):
            return Denied(code, reason)
    if ENV_SENSITIVE.search(text) and DISCLOSE_VERB.search(n):
        return Denied("sensitive_file_read", "citanje .env fajla")
    return None


# ---------------------------------------------------------------------------
# entry points
# ---------------------------------------------------------------------------
_SUBST = re.compile(r"\$\(([^()]*)\)|`([^`]*)`|<\(([^()]*)\)")


def check_structural(text):
    """Return a :class:`Denied` for a destructive/disclosure *structure*, else None."""
    t = str(text or "")
    if _FORK_BOMB.search(t):
        return Denied("fork_bomb", "fork bomba ugrozava dostupnost sistema")
    d = _check_sql(t)
    if d:
        return d
    for rx in _RCE_RAW:
        if rx.search(t):
            return Denied("remote_code_exec",
                          "preuzeti sadrzaj se izvrsava u shellu")
    for m in _SUBST.finditer(t):
        inner = next((g for g in m.groups() if g is not None), "")
        if inner:
            d = check_structural(inner)
            if d:
                return d
    for stmt in _split_top(t, ";&\n"):
        stages = _split_top(stmt, "|", pairs=())
        for st in stages:
            d = _check_command(st)
            if d:
                return d
        d = _check_remote_pipe(stages)
        if d:
            return d
    return _check_nl(t)


def check(text):
    """Full deny decision: structural parse first, then the policy pre-filter."""
    return check_structural(text) or check_policy(text)
