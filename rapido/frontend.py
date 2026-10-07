"""Rápido text frontend: written Brazilian Portuguese in, text in the model's own alphabet out.

Numbers, money, percentages, ordinals, times, dates and common abbreviations are read aloud in
pt-BR. `modernize()` maps pre-1943 spellings found in the training audiobooks (elle, janella,
pharmacia, inglez...) to modern ones; the reform changed spelling, not pronunciation. Nothing here
raises on text.
"""
import re
import unicodedata

from num2words import num2words

LETTERS = "abcdefghijklmnopqrstuvwxyzáàâãçéêíóôõúü"
PUNCT = ".,?!;:-'"
VOCAB = ["<pad>", "<unk>", " "] + list(LETTERS) + list(PUNCT)

TYPOGRAPHY = str.maketrans({"’": "'", "‘": "'", "´": "'", "`": "'", "“": '"', "”": '"', "„": '"', "«": '"',
                            "»": '"', "–": "-", "—": "-", "−": "-", "…": "...", " ": " "})
MONTHS = ["janeiro", "fevereiro", "março", "abril", "maio", "junho", "julho", "agosto", "setembro", "outubro",
          "novembro", "dezembro"]
ABBREV = {
    "sr.": "senhor", "sra.": "senhora", "srta.": "senhorita", "dr.": "doutor", "dra.": "doutora",
    "prof.": "professor", "profa.": "professora", "sto.": "santo", "sta.": "santa", "av.": "avenida",
    "r.": "rua", "etc.": "etcétera", "ex.": "exemplo", "obs.": "observação", "pág.": "página", "p.": "página",
    "tel.": "telefone", "n.º": "número", "nº": "número", "no.": "número", "vs.": "versus", "aprox.": "aproximadamente",
}
UNITS = {"km": "quilômetros", "kg": "quilos", "g": "gramas", "m": "metros", "cm": "centímetros", "mm": "milímetros",
         "l": "litros", "ml": "mililitros", "h": "horas", "min": "minutos", "s": "segundos", "km/h": "quilômetros por hora",
         "°c": "graus celsius", "°": "graus", "gb": "gigabytes", "mb": "megabytes"}
FEM = {"um": "uma", "dois": "duas"}


def n2w(n, **kw):
    try:
        return num2words(n, lang="pt_BR", **kw).replace(",", "")
    except Exception:
        return " ".join(num2words(int(d), lang="pt_BR") for d in str(n) if d.isdigit())


def fem(words):
    """'um', 'dois', 'duzentos'... -> feminine, for horas, pessoas-style nouns."""
    out = []
    for w in words.split():
        w = FEM.get(w, w)
        w = re.sub(r"entos$", "entas", w) if w.endswith("entos") and w != "quinhentos" else w
        out.append("quinhentas" if w == "quinhentos" else w)
    return " ".join(out)


def parse_num(s):
    """'1.250.000' -> 1250000, '3,5' -> 3.5, '1.250,75' -> 1250.75"""
    s = s.strip().rstrip(".,")
    if re.fullmatch(r"\d{1,3}(\.\d{3})+(,\d+)?", s):
        s = s.replace(".", "")
    return float(s.replace(",", ".")) if "," in s else int(s)


def money(m):
    v = parse_num(m.group(1))
    reais, cent = int(v), round((v - int(v)) * 100) if isinstance(v, float) else 0
    out = []
    if reais:
        out.append(n2w(reais) + (" de" if reais >= 1_000_000 and reais % 1_000_000 == 0 else "") +
                   (" real" if reais == 1 else " reais"))
    if cent:
        out.append(n2w(cent) + (" centavo" if cent == 1 else " centavos"))
    return " " + (" e ".join(out) if out else "zero reais") + " "


def time_(m):
    h = int(m.group(1))
    mi = int(m.group(2) or 0) if m.re.groups > 1 else 0
    if h > 24 or mi > 59:
        return m.group(0)
    hs = fem(n2w(h)) + (" hora" if h == 1 else " horas")
    return " " + hs + (f" e {fem(n2w(mi)) if mi in (1, 2) else n2w(mi)} minuto{'s' if mi != 1 else ''}" if mi else "") + " "


def date(m):
    d, mo, y = int(m.group(1)), int(m.group(2)), m.group(3)
    if not (1 <= d <= 31 and 1 <= mo <= 12):
        return m.group(0)
    day = "primeiro" if d == 1 else n2w(d)
    out = f" {day} de {MONTHS[mo - 1]}"
    if y:
        y = int(y) + (2000 if len(y) == 2 and int(y) < 50 else 1900 if len(y) == 2 else 0)
        out += f" de {n2w(y)}"
    return out + " "


def number(m):
    s = m.group(0)
    if re.fullmatch(r"\d{10,}", s):  # phone numbers, codes: digit by digit
        return " " + " ".join(n2w(int(c)) for c in s) + " "
    v = parse_num(s)
    if isinstance(v, float):
        whole, frac = s.replace(".", "").split(",")
        return f" {n2w(int(whole))} vírgula {' '.join(n2w(int(c)) for c in frac) if frac.startswith('0') else n2w(int(frac))} "
    return f" {n2w(v)} "


ROMAN = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}
ROMAN_CTX = r"(século|séculos|capítulo|capítulos|volume|tomo|parte|ato|livro|papa|rei|rainha|dom|dona|pio|joão|pedro|luís|henrique|elizabeth|guerra mundial|bienal|edição|congresso|fórum)"
FEM_NOUNS = {"pessoa", "pessoas", "hora", "horas", "criança", "crianças", "vez", "vezes", "semana", "semanas",
             "mulher", "mulheres", "cidade", "cidades", "casa", "casas", "vaga", "vagas", "família", "famílias",
             "empresa", "empresas", "escola", "escolas", "página", "páginas", "moeda", "moedas", "unidade",
             "unidades", "tonelada", "toneladas", "pessoa", "noite", "noites", "mãe", "mães", "filha", "filhas",
             "aluna", "alunas", "garrafa", "garrafas", "xícara", "xícaras", "colher", "colheres", "fruta", "frutas",
             "árvore", "árvores", "rua", "ruas", "vida", "vidas", "morte", "mortes", "parte", "partes", "linha",
             "linhas", "palavra", "palavras", "questão", "questões", "região", "regiões", "nação", "nações",
             "edição", "edições", "medalha", "medalhas", "gramas" }
MASC_A = {"dia", "dias", "mapa", "mapas", "problema", "problemas", "sistema", "sistemas", "programa", "programas",
          "tema", "temas", "clima", "planeta", "planetas", "idioma", "idiomas", "poema", "poemas", "telefonema",
          "cinema", "cinemas", "atleta", "atletas", "jornalista", "artista", "artistas", "turista", "turistas"}


def roman_to_int(r):
    total, prev = 0, 0
    for ch in reversed(r):
        v = ROMAN[ch]
        total = total - v if v < prev else total + v
        prev = max(prev, v)
    return total


def romans(t):
    """Roman numerals after context words (século XVIII, Dom Pedro II) and standalone II..XXXIX."""
    def ctx(m):
        n = roman_to_int(m.group(2))
        word = m.group(1).lower()
        if word in ("papa", "rei", "rainha", "dom", "dona", "pio", "joão", "pedro", "luís", "henrique", "elizabeth") and n <= 10:
            return f"{m.group(1)} {n2w(n, to='ordinal')}"
        return f"{m.group(1)} {n2w(n)}"
    t = re.sub(r"(?i)\b" + ROMAN_CTX + r"\s+([IVXLCDM]{1,7})\b", ctx, t)
    return t


def is_fem(word):
    w = word.lower().strip(".,;:!?")
    if w in MASC_A:
        return False
    return w in FEM_NOUNS or (len(w) > 3 and (w.endswith("as") or w.endswith("a")) and not w.endswith("ma"))


def agree(m):
    """Cardinal + feminine noun: 'duas pessoas', 'duzentas horas', 'uma vez'."""
    num, noun = m.group(1), m.group(2)
    if "," in num:
        return m.group(0)
    v = parse_num(num)
    words = n2w(v)
    if is_fem(noun):
        words = fem(words)
    return f" {words} {noun}"


def phones(t):
    """(11) 98765-4321, 0800 721 3344 -> digits/pairs, not big cardinals."""
    def rd(g):
        g = re.sub(r"\D", "", g)
        return " ".join(n2w(int(c)) for c in g)
    t = re.sub(r"\((\d{2})\)\s*(\d{4,5})[-\s]?(\d{4})", lambda m: f" {rd(m.group(1))}, {rd(m.group(2))}, {rd(m.group(3))} ", t)
    t = re.sub(r"\b(0800|0300|0500|0900)[\s-]?(\d{3})[\s-]?(\d{4})\b", lambda m: f" {rd(m.group(1))}, {rd(m.group(2))}, {rd(m.group(3))} ", t)
    t = re.sub(r"\b(\d{4,5})-(\d{4})\b", lambda m: f" {rd(m.group(1))}, {rd(m.group(2))} ", t)
    return t


def spoken(text):
    t = " " + text.translate(TYPOGRAPHY) + " "
    t = phones(t)
    t = romans(t)
    t = re.sub(r"R\$\s*(\d{1,3}(?:\.\d{3})+(?:,\d+)?|\d+(?:,\d+)?)", money, t)
    t = re.sub(r"(\d{1,3}(?:\.\d{3})+(?:,\d+)?|\d+(?:,\d+)?)\s*%", lambda m: f" {number(re.match(r'.+', m.group(1)))} por cento ", t)
    t = re.sub(r"\b(\d{1,2})/(\d{1,2})(?:/(\d{2}|\d{4}))?\b", date, t)
    t = re.sub(r"\b(\d{1,2})(?::|h)(\d{2})\b(?:min)?", time_, t)
    t = re.sub(r"\b(\d{1,2})h\b", time_, t)
    t = re.sub(r"(?i)(?<![\w.])(" + "|".join(re.escape(k) for k in sorted(ABBREV, key=len, reverse=True)) + r")(?=\s)",
               lambda m: ABBREV[m.group(1).lower()], t)
    t = re.sub(r"(\d{1,3}(?:\.\d{3})+(?:,\d+)?|\d+(?:,\d+)?)\s*(km/h|km|kg|mm|cm|ml|min|gb|mb|°\s*c|°|g|m|l|s|h)\b",
               lambda m: m.group(1) + " " + unit_word(m.group(2)), t, flags=re.I)
    t = re.sub(r"(\d+)\s*º", lambda m: f" {n2w(int(m.group(1)), to='ordinal')} ", t)
    t = re.sub(r"(\d+)\s*ª", lambda m: f" {fem_ord(n2w(int(m.group(1)), to='ordinal'))} ", t)
    t = re.sub(r"(?<![\d.,])(\d{1,3}(?:\.\d{3})+|\d+)\s+([A-Za-zÀ-ú]+)", agree, t)
    t = re.sub(r"\d{1,3}(?:\.\d{3})+(?:,\d+)?|\d+(?:,\d+)?", number, t)
    t = t.replace("&", " e ").replace("@", " arroba ").replace("+", " mais ").replace("=", " igual a ")
    return t


def unit_word(u):
    return UNITS[re.sub(r"\s", "", u.lower())]


def fem_ord(w):
    return " ".join(x[:-1] + "a" if x.endswith("o") else x for x in w.split())


ARCHAIC = [
    (r"ph", "f"), (r"th", "t"), (r"(?<=[a-z])y(?=[a-z])", "i"),
    (r"ll", "l"), (r"mm", "m"), (r"nn", "n"), (r"tt", "t"), (r"pp", "p"), (r"ff", "f"), (r"bb", "b"), (r"dd", "d"),
    (r"gg", "g"), (r"cc(?=[aou])", "c"),
]
WORDS = {"hontem": "ontem", "emquanto": "enquanto", "emfim": "enfim", "sahir": "sair", "cahir": "cair",
         "ás": "às", "á": "à", "sómente": "somente", "egreja": "igreja", "logar": "lugar", "comtudo": "contudo",
         "emfim": "enfim", "inglez": "inglês", "portuguez": "português", "francez": "francês", "mez": "mês",
         "vez": "vez", "pez": "pés", "fóra": "fora", "côr": "cor", "poz": "pôs", "poz-se": "pôs-se", "fez-se": "fez-se", "ella": "ela", "elle": "ele", "elles": "eles",
         "ellas": "elas", "n'um": "num", "n'uma": "numa", "d'isto": "disto", "d'isso": "disso", "d'aqui": "daqui",
         "d'ali": "dali", "d'elle": "dele", "d'ella": "dela", "d'esse": "desse", "d'essa": "dessa", "d'este": "deste",
         "d'esta": "desta", "d'aquelle": "daquele", "d'aquella": "daquela"}


def modernize(text):
    """Pre-reform spellings -> modern pt-BR, word by word (training text only)."""
    def fix(w):
        lw = w.lower()
        core = lw.strip(PUNCT + '"')
        if core in WORDS:
            return lw.replace(core, WORDS[core])
        for a, b in ARCHAIC:
            lw = re.sub(a, b, lw)
        return lw
    return " ".join(fix(w) for w in text.split())


def alphabet(text):
    text = unicodedata.normalize("NFC", text.lower())
    out = []
    for ch in text:
        if ch in LETTERS or ch in PUNCT or ch == " ":
            out.append(ch)
        elif ch in "\"()[]{}":
            out.append(" ")
        else:
            base = "".join(c for c in unicodedata.normalize("NFKD", ch) if not unicodedata.combining(c))
            out.append(base if base and all(c in LETTERS for c in base) else " ")
    t = re.sub(r"\s+", " ", "".join(out))
    t = re.sub(r"\s+([.,?!;:])", r"\1", t)  # no space before punctuation
    t = re.sub(r"([.,?!;:])\1+", r"\1", t).replace("..", ".")
    return t.strip(" -'")


def normalize(text, archaic=False):
    try:
        text = spoken(text)
    except Exception:  # keep the words rather than lose the sentence
        pass
    if archaic:
        text = modernize(text)
    return alphabet(text)


def encode(text):
    stoi = {c: i for i, c in enumerate(VOCAB)}
    return [stoi.get(c, 1) for c in text]


if __name__ == "__main__":
    tests = [
        "Olá! O pedido custa R$ 1.250,50 e chega em 15/10/2026 às 14:30.",
        "A inflação foi de 4,5% em 2025; o 1º lugar ficou com a 2ª colocada.",
        "O Sr. Silva mora na Av. Paulista, nº 1000, a 12 km daqui.",
        "Ligue para 11987654321 às 9h. Temperatura: 23°C.",
        "Ella abriu a janella e poz-se a mirar o theatro emquanto sahia.",
        "Você vem amanhã? Sim, chego às 2h com 3 amigos.",
    ]
    for t in tests:
        print(t, "\n ->", normalize(t, archaic=True), "\n")
