"""
JORO v1.6.0 - motera chess miorina amin'ny 3 fototra:
  1) King safety    (is my king safe? is his king safe?)
  2) Attack compare (allow / care)
  3) Look the future (ghost board + pattern checkmate + sacrifice cost)
Search: iterative deepening + alpha-beta + quiescence.
v0.2: pattern checkmate matanjaka + opening book.
v0.3: ghost mate ho an'ny mpanohitra (defense), decision IGNORE/ALLOW/CARE,
      search maivana (gating + cache + lazy eval).
v0.4: plan_needs = open file/diagonal, sacrifice, pressure, block square
      (vokatry ny ghost) + enable_gain: ALLOW raha manala ny blocker ny attack-ny.
v0.5: endgame = passed pawn / promotion race vs king hunt, king activity,
      opposition, mop-up (drive king to edge), depth +1, repetition.
v0.6: ghost aorian'ny promotion (queen vaovao vs king-ko) + KBN corner.
v0.7: ghost mijery KING POSITION hafa (target), defense mety tsy hita (hidden),
      smart promotion (Q/R/B/N) ho an'ny checkmate, forced checks.
v0.8: FORCING MOVES (check, check-sacrifice, sacrifice manala blocker, mate threat),
      mampidi-bidy ny vidin'ny fanerena ny king mankany amin'ny target; check extension.
v0.8.1: back-rank/smothered mate (piece ao anatiny manakana ny king) hita sy hamarinina
      amin'ny tena board; decide tsy milaza 'haingana' raha samy lavitra; filtra fanampiny.
v0.8.2: mode selfplay (JORO vs JORO) + version aseho (hamarinana ny rakitra ampiasaina).
v0.8.3: FIX bug timeout: ny board dia averina amin'ny toerana tany am-boalohany
        rehefa tapaka ny search (tsy misy move mijanona ao anaty board intsony).
v0.8.4: ny FEN dia diovina (esorina ny ':' na quote eo am-piandohana, toy ny diso fampidirana).
v0.8.5: KBN: king-ko manatona (x4), famerana ny escape ny king mpanohitra, search lalina
        kokoa rehefa <=4 ny piece (hampiasa tsara ny fotoana).
v0.9.0: W-MANEUVER: K+B+N vs K voavaha tanteraka (retrograde, mate indrindra 33 move);
        python joro.py buildkbn  -> kbnk.dat ; ampiasain'ny JORO ho vidin'ny fanerena.
v1.0.0: SEARCH + EVAL MAIVANA: transposition table, killer/history, null move, LMR,
        delta pruning, famerana ny sosona ghost (tier 0 raha tsy misy alert), PST
        voatahiry, positional terms (bishop pair, rook file, pawn structure, king),
        mampiasa ny fotoana (depth hatramin'ny 10), natao ampiasaina ny partial iteration.
v1.0.1: FEN 'startpos' = toerana fiandohana.
v1.0.2: endgame (rook, sns.) tsy mampiasa ghost feno foana intsony (nps x3-5);
        ghost feno rehefa alert / pawn akaiky promotion / <=6 ny piece.
v1.1.0: mode MATCH: JORO vs Stockfish (UCI_Elo voafetra) -> W/D/L + Elo mombamomba.
v1.2.0: A/B test: 'match --opp lite' = JORO feno vs JORO-lite (tsy misy sosona ghost/forcing)
        mba hanombana raha tena mahasoa ny hevitry ny JORO. Mety amin'ny PyPy (haingana kokoa).
v1.3.0: VOKATRA A/B: JORO feno 1-28 vs lite => ny sosona ghost (lanja feno) dia MAMPIVERINA.
        Ankehitriny: lanja voafetra (--joro-weight, default 0.25) + fetra (--joro-cap 80)
        ary --nolayers (lite) ho an'ny match vs Stockfish, mba hitady lanja mahasoa.
v1.4.0: OPENING mifototra amin'ny tanjona (tsy book intsony): fivoarana tsy mampiaraka,
        fahefana afovoany (d/e), fanakanana ny king mpanohitra tsy hicastle (ghost: path
        castling, file/diagonal, sacrifice pawn). Vita ny opening -> miverina ny search + ghost.
        --opening joro|book|none ; --lite-opening none|book.
v1.4.1: opening: sazy ny pawn sisiny (a/h) mandroso aloha, bonus castling ho an'ny king-ko,
        sazy raha very ny castling right tsy nicastle; --seed ho an'ny match (matrix parallel).
v1.4.2: statistika hafainganam-pikarohana (nps, depth) ao amin'ny match mba hizarana ny
        vidin'ny fotoana amin'ny kalitaon'ny evaluation.
v1.5.0: VOKATRA D/E: ghost per-node = 7x miadana (nps 3 600 vs 26 400), fa mitovy ny haavony amin'ny
        depth mitovy. => JORO lasa MPANDINIKA ROOT: search maivana (lite) no mitady, ary ny ghost/
        target/threat dia ampiasaina amin'ny 4 move voalohany ihany (--joro-mode root, default).
v1.6.0: ENGINE C++ (joro.cpp): bitboard, perft voamarina, ~3 Mnps (100x PyPy). Ampiasaina ao amin'ny
        match amin'ny '--cpp ./joro_cpp' (UCI). g++ -O3 -std=c++17 -o joro_cpp joro.cpp

Fampiasana:  pip install chess
  python joro.py play white
  python joro.py play black --book gm.bin     (Polyglot .bin, tsy voatery)
  python joro.py analyze "<FEN>"
  python joro.py selfplay "<FEN>" --plies 70 --time 2   (JORO milalao irery)
  python joro.py buildkbn                       (mamorona kbnk.dat, mila numpy)
  python joro.py match --games 4 --sf-elo 1500 --time 2   (vs Stockfish, mila apt install stockfish)
  python joro.py match --opp lite --games 20 --time 2      (JORO feno vs JORO-lite: A/B test)
  Safidy: --depth 4  --time 5  --nobook
"""
import argparse
import os
import zlib
from collections import namedtuple
import random
import time
import chess
import chess.polyglot
from chess import PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING, WHITE, BLACK

MATE = 100000
VERSION = "1.6.0"
VAL = {PAWN: 100, KNIGHT: 320, BISHOP: 330, ROOK: 500, QUEEN: 900, KING: 0}
ATT_W = {PAWN: 1, KNIGHT: 2, BISHOP: 2, ROOK: 3, QUEEN: 5, KING: 0}

# ---------------------------------------------------------------- Patterns
# (anarana, piece ilaina (ankoatra ny king), lanjany = fahamoran'ny checkmate)
PATTERNS = [
    ("Queen+King", {QUEEN: 1}, 1.2),
    ("Queen+Rook", {QUEEN: 1, ROOK: 1}, 1.3),
    ("Queen+Bishop", {QUEEN: 1, BISHOP: 1}, 1.1),
    ("Queen+Knight", {QUEEN: 1, KNIGHT: 1}, 1.1),
    ("Rook+King", {ROOK: 1}, 1.0),
    ("Rook+Rook", {ROOK: 2}, 1.3),
    ("Rook+Knight+King", {ROOK: 1, KNIGHT: 1}, 1.0),
    ("Rook+Bishop+King", {ROOK: 1, BISHOP: 1}, 1.0),
    ("Double Knight+King", {KNIGHT: 2}, 0.8),
    ("Bishop+Knight+King", {BISHOP: 1, KNIGHT: 1}, 0.9),
    ("Double Bishop+King", {BISHOP: 2}, 0.8),
    ("Double Bishop+Knight+King", {BISHOP: 2, KNIGHT: 1}, 1.0),
    ("Pawn+Pawn+King", {PAWN: 2}, 0.7),
    ("Pawn+King+Rook", {PAWN: 1, ROOK: 1}, 1.0),
    ("Pawn+King+Bishop/Knight", {PAWN: 1, KNIGHT: 1}, 0.9),
    ("Pawn+King+Queen", {PAWN: 1, QUEEN: 1}, 1.1),
]


# ---------------------------------------------------------------- helpers
def dist(a, b):
    return max(abs(chess.square_file(a) - chess.square_file(b)),
               abs(chess.square_rank(a) - chess.square_rank(b)))


def king_zone(ksq):
    return chess.SquareSet(chess.BB_KING_ATTACKS[ksq] | chess.BB_SQUARES[ksq])


def ghost_board(board, color, keep_blockers=False):
    """Piece-n'i `color` + king mpanohitra ihany (tsy misy defense).
    keep_blockers=True: tehirizina koa ny piece mpanohitra manodidina ny king
    (cas exceptional: smothered mate, Knight + piece mpanohitra, sns.)."""
    g = chess.Board(None)
    eks = board.king(not color)
    for sq, p in board.piece_map().items():
        if p.color == color or p.piece_type == KING:
            g.set_piece_at(sq, p)
        elif keep_blockers and eks is not None and dist(sq, eks) == 1:
            g.set_piece_at(sq, p)
    g.turn = not color
    g.castling_rights = 0
    return g


# ---------------------------------------------------------------- 1) King safety
def king_safety(board, color):
    ksq = board.king(color)
    if ksq is None:
        return 0
    zone = chess.SquareSet(chess.BB_KING_ATTACKS[ksq])
    defenders = sum(1 for s in zone
                    if board.color_at(s) == color and board.piece_type_at(s) != KING)
    shield = 0
    step = 8 if color == WHITE else -8
    for df in (-1, 0, 1):
        f = chess.square_file(ksq) + df
        if 0 <= f <= 7:
            s = ksq + step + df
            if 0 <= s < 64 and board.piece_at(s) == chess.Piece(PAWN, color):
                shield += 1
    open_files = 0
    for df in (-1, 0, 1):
        f = chess.square_file(ksq) + df
        if 0 <= f <= 7:
            if not any(board.piece_at(chess.square(f, r)) == chess.Piece(PAWN, color)
                       for r in range(8)):
                open_files += 1
    score = defenders * 6 + shield * 8 - open_files * 12
    return score if board.pieces(QUEEN, not color) else score // 2


# ---------------------------------------------------------------- 2) Attack
def attack_power(board, color):
    eks = board.king(not color)
    if eks is None:
        return 0
    zone = chess.BB_KING_ATTACKS[eks] | chess.BB_SQUARES[eks]
    total, n = 0, 0
    for sq in chess.SquareSet(board.occupied_co[color] & ~board.kings):
        c = _pc(board.attacks_mask(sq) & zone)
        if c:
            total += ATT_W[board.piece_type_at(sq)] * c
            n += 1
    return total * min(n, 4)


def care_factor(mine, theirs):
    """0 = tsy raharahiana, 0.3 = allowed, 1 = mitandrina."""
    if theirs < 12:
        return 0.0
    if mine >= theirs * 1.3:
        return 0.3
    return 1.0


# ---------------------------------------------------------------- 3) Future
def pawn_future(board, color, eks):
    """Smart promote + rule of the square (ghost: king mpanohitra irery)."""
    total = 0
    for s in board.pieces(PAWN, color):
        f, r = chess.square_file(s), chess.square_rank(s)
        steps = 7 - r if color == WHITE else r
        if steps > 3:
            continue
        step = 8 if color == WHITE else -8
        if any(board.piece_at(s + step * i) for i in range(1, steps + 1)):
            continue                                  # voasakana
        promo = chess.square(f, 7 if color == WHITE else 0)
        b = (4 - steps) * 8
        if chess.BB_KNIGHT_ATTACKS[promo] & chess.BB_SQUARES[eks]:
            b += 10                                   # promote Knight = check
        elif promo != eks and chess.BB_RAYS[promo][eks]:
            b += 6                                    # promote Queen/Rook = check
        enemy_to_move = board.turn != color
        moves = steps if enemy_to_move else steps - 1
        if dist(eks, promo) - 1 > moves:
            b += 25                                   # tsy azon'ny king tsaraina
        total += b
    return total


def pattern_info(board, color, eks, g):
    """Mamerina (bonus, anaran'ny pattern tsara indrindra)."""
    mine = {t: list(board.pieces(t, color)) for t in (PAWN, KNIGHT, BISHOP, ROOK, QUEEN)}
    ksq = board.king(color)
    fe, re = chess.square_file(eks), chess.square_rank(eks)
    edge = min(fe, 7 - fe, re, 7 - re)
    edge_factor = 1 + (3 - edge) * 0.25               # king eo an-jorony = mora kokoa
    best, name = 0.0, ""
    for pname, need, w in PATTERNS:
        if any(len(mine[t]) < n for t, n in need.items()):
            continue
        ds = [dist(ksq, eks)]
        for t, n in need.items():
            ds += sorted(dist(s, eks) for s in mine[t])[:n]
        val = max(0.0, 8 - sum(ds) / len(ds)) * 3 * w * edge_factor
        if val > best:
            best, name = val, pname
    extra, strong = 0, 0
    zone = chess.SquareSet(chess.BB_KING_ATTACKS[eks])
    for s in mine[ROOK] + mine[QUEEN]:
        fd = abs(chess.square_file(s) - fe)
        rd = abs(chess.square_rank(s) - re)
        if fd == 1 or rd == 1:
            extra += 5                                # cut-off (manetsika ny king)
        if len(g.attacks(s) & zone) >= 2:
            strong += 1
    if strong >= 2:
        extra += 20                                   # ladder mate
        name = name or "Ladder (heavy pieces)"
    total = int(best) + min(extra, 35) + pawn_future(board, color, eks)
    if not name and pawn_future(board, color, eks) > 0:
        name = "Pawn+Smart promote"
    return total, name


Threat = namedtuple("Threat", "score tempo cost name mate")
_THREAT_CACHE = {}


def _pos_key(board):
    return (board.occupied_co[WHITE], board.occupied_co[BLACK], board.pawns,
            board.knights, board.bishops, board.rooks, board.queens,
            board.kings, board.turn)


def _mates_in_1(g, color):
    """Checking moves ihany no andramana (tsy ny move rehetra) -> maivana."""
    res = []
    for m in list(g.legal_moves):
        if not g.gives_check(m):
            continue
        g.push(m)
        if g.is_checkmate():
            res.append(m)
        g.pop()
    return res


def ghost_mate_in_1(board, color, blockers=False):
    g = ghost_board(board, color, keep_blockers=blockers)
    if g.is_check():
        return [None] if g.is_checkmate() else []
    g.turn = color
    return _mates_in_1(g, color)


# ---------------------------------------------------------------- Ghost v2
# 1) King mpanohitra : tsy ny toerana ankehitriny ihany, fa ny toerana mety rehetra
#    akaiky (<=2 dingana) izay mety hanamora ny future -> "target".
# 2) Defense         : tsy ny hita ihany; ny piece afaka mankany amin'ny toerana
#    manan-danja ao anatin'ny 1 move (tsy ampoizina) dia voaisa koa.
# 3) Smart promotion : Q/R/B/N, jerena izay manatontosa ny checkmate.
Target = namedtuple("Target", "square net value free dist hidden promo tempo mate0 incheck by")
_TARGET_CACHE = {}


def _pc(x):
    return bin(x).count("1")


def _edge_dist(sq):
    f, r = chess.square_file(sq), chess.square_rank(sq)
    return min(f, 7 - f, r, 7 - r)


def defense_potential(board, color, T):
    """Defense mety hitranga manoloana ny king mpanohitra raha eo amin'ny T:
    piece mpanohitra afaka mankany (1 move) amin'ny toerana manan-danja
    (manodidina ny king, tsipika-check, toeran'ny attaquant) = 1.0;
    ny akaiky (<=3) saingy tsy mahatratra = 0.4. Tsy miankina amin'ny
    zavatra hitan'ny maso fotsiny."""
    zone = chess.BB_KING_ATTACKS[T] | chess.BB_SQUARES[T]
    crit = zone
    for sq in chess.SquareSet(board.occupied_co[color] & ~board.kings):
        mask = board.attacks_mask(sq)
        if mask & zone:
            crit |= chess.BB_SQUARES[sq]
            for z in chess.SquareSet(mask & zone):
                crit |= chess.between(sq, z)
    own = board.occupied_co[not color]
    h = 0.0
    for e in chess.SquareSet(own & ~board.kings):
        p = board.piece_at(e)
        reach = board.attacks_mask(e) & ~own
        if p.piece_type == PAWN:
            push = e + (8 if p.color == WHITE else -8)
            if 0 <= push < 64 and not board.piece_at(push):
                reach |= chess.BB_SQUARES[push]
        if reach & crit:
            h += 1.0
        elif p.piece_type != PAWN and dist(e, T) <= 3:
            h += 0.4
    return min(h, 6.0)


def _att_union(gb):
    a = 0
    for sq in gb.piece_map():
        a |= gb.attacks_mask(sq)
    return a


def future_targets(board, color):
    """Lisitry Target (tsara indrindra aloha): toerana mety ho misy ny king
    mpanohitra (anisan'izany ny toerana ankehitriny) + smart promotion."""
    key = (_pos_key(board), color)
    r = _TARGET_CACHE.get(key)
    if r is not None:
        return r
    out = []
    eks, ksq = board.king(not color), board.king(color)
    if eks is not None and ksq is not None and (board.occupied_co[color] & ~board.kings):
        g = chess.Board(None)
        for sq, p in board.piece_map().items():
            if p.color == color:
                g.set_piece_at(sq, p)
        variants = [(_att_union(g), None, 0, 0)]
        step = 8 if color == WHITE else -8
        for s in board.pieces(PAWN, color):
            steps = promo_steps(s, color)
            if steps > 3:
                continue
            path = [s + step * i for i in range(1, steps + 1)]
            if any(board.color_at(p) == color for p in path):
                continue
            extra = steps * 8 + (10 if any(board.piece_at(p) for p in path) else 0)
            for pt in (QUEEN, ROOK, BISHOP, KNIGHT):
                g2 = g.copy(stack=False)
                g2.remove_piece_at(s)
                g2.set_piece_at(path[-1], chess.Piece(pt, color))
                variants.append((_att_union(g2), pt, steps, extra))
        cands = [t for t in range(64)
                 if dist(t, eks) <= 2 and dist(t, ksq) > 1
                 and not (board.occupied_co[color] & chess.BB_SQUARES[t])
                 and (t == eks or board.piece_at(t) is None)]
        pool = []
        for att, pt, steps, extra in variants:
            for T in cands:
                zm = chess.BB_KING_ATTACKS[T]
                n = _pc(zm)
                cov = _pc(zm & att)
                free = n - cov
                full = free == 0
                incheck = bool(att & chess.BB_SQUARES[T])
                mate0 = incheck and full
                val = (cov * 12 + (40 if full else 0) + (15 if incheck else 0)
                       + (150 if mate0 else 0) + (3 - _edge_dist(T)) * 5)
                d = dist(eks, T)
                force = d * 8 + (10 if d > 0 and board.is_attacked_by(color, T) else 0) + extra
                pool.append((val - force, val, force, T, pt, d, free, full, mate0, incheck, steps, extra))
        pool.sort(key=lambda x: -x[0])
        hcache = {}
        forcing = None
        for i, (pre, val, force, T, pt, d, free, full, mate0, incheck, steps, extra) in enumerate(pool[:12]):
            H = 0.0
            if i < 3 and val >= 60:                       # defense mety: ho an'ny tsara ihany
                if T not in hcache:
                    hcache[T] = defense_potential(board, color, T)
                H = hcache[T]
            by = None
            if d > 0 and val >= 60:                       # forcing move: check / sacrifice
                if forcing is None:
                    forcing = forcing_moves(board, color)
                best_c = None
                for f in forcing:
                    if T in f.dests:
                        c = (4 if f.only else 8) + f.cost / 10.0
                        if best_c is None or c < best_c:
                            best_c, by = c, f.san
                if best_c is not None and best_c + extra < force:
                    force = best_c + extra                # vidiny mora kokoa
                else:
                    by = None
            net = val / (1 + 0.25 * H) - force
            tempo = min(9, (1 if by else d) + steps + (1 if full else 3))
            out.append(Target(T, net, val, free, d, H,
                              chess.piece_symbol(pt).upper() if pt else None,
                              tempo, mate0, incheck, by))
        out.sort(key=lambda t: -t.net)
    if len(_TARGET_CACHE) > 200000:
        _TARGET_CACHE.clear()
    _TARGET_CACHE[key] = out
    return out


Forcing = namedtuple("Forcing", "move san kind cost dests only")


def forcing_moves(board, color, deep=False):
    """Move manery ny mpanohitra (ny ahy, `color`):
      CHECK       : check azo ekena, manery ny king (dests = toerana azony alehany)
      CHECK_SAC   : check izay azon'ny mpanohitra alain'ny piece (sacrifice, cost = very)
      SACRIFICE   : capture blocker voaro (vokatry ny plan_needs) lafo kokoa ny piece-ko
      CAPTURE     : capture blocker tsy lafo
      MATE_THREAT : (deep) move mangina izay mampahatahotra mate-in-1 raha tsy valiana
    only=True raha ny king ihany no afaka mamaly (tsy misy valiny hafa)."""
    b = board.copy(stack=False)
    b.turn = color
    eks = b.king(not color)
    if eks is None or b.is_attacked_by(color, eks):
        return []
    blockers = {n.square: n for n in plan_needs(board, color) if n.kind.startswith("open")}
    base_mate = False
    if deep:                                   # mate-in-1 efa misy ve? (tsy ilaina ny mate threat)
        for m2 in list(b.legal_moves):
            if b.gives_check(m2):
                b.push(m2)
                base_mate = b.is_checkmate()
                b.pop()
                if base_mate:
                    break
    deep_threat = deep and not base_mate
    res = []
    for m in list(b.legal_moves):
        gives = b.gives_check(m)
        take = m.to_square in blockers and b.is_capture(m)
        if not (gives or take or deep_threat):
            continue
        san = b.san(m)
        mover = b.piece_type_at(m.from_square)
        b.push(m)
        if gives:
            replies = list(b.legal_moves)
            ek = b.king(not color)
            dests = [r.to_square for r in replies if r.from_square == ek]
            only = bool(replies) and all(r.from_square == ek for r in replies)
            kind, cost = "CHECK", 0
            caps = [r for r in replies if r.to_square == m.to_square]
            if caps:
                cap_val = min(VAL[b.piece_type_at(r.from_square)] for r in caps)
                kind = "CHECK_SAC"
                cost = (VAL[mover] if not b.is_attacked_by(color, m.to_square)
                        else max(0, VAL[mover] - cap_val))
            res.append(Forcing(m, san, kind, cost, dests, only))
        elif take:
            n = blockers[m.to_square]
            res.append(Forcing(m, san, n.action, n.cost, [], False))
        elif deep_threat:
            b.push(chess.Move.null())
            threat = False
            for m2 in list(b.legal_moves):
                if b.gives_check(m2):
                    b.push(m2)
                    threat = b.is_checkmate()
                    b.pop()
                    if threat:
                        break
            b.pop()
            if threat:
                res.append(Forcing(m, san, "MATE_THREAT", 0, [], False))
        b.pop()
    res.sort(key=lambda f: (0 if f.only else 1, f.cost))
    return res


def forced_destinations(board, color):
    """[(san, [squares], tsy-misy-valiny-hafa)] ho an'ny check manery ny king."""
    return [(f.san, f.dests, f.only) for f in forcing_moves(board, color) if f.dests]


def _real_mate_in_1(board, color, moves):
    """Hamarino amin'ny TENA board raha tena checkmate ireo move avy amin'ny ghost
    (back-rank mate, smothered mate...)."""
    b = board.copy(stack=False)
    b.turn = color
    eks = b.king(not color)
    if eks is None or b.is_attacked_by(color, eks):
        return False
    for m in moves:
        if m is not None and m in b.legal_moves:
            b.push(m)
            ok = b.is_checkmate()
            b.pop()
            if ok:
                return True
    return False


def threat_profile(board, color):
    """Ghost: ny piece an'i `color` vs king mpanohitra.
    Tier 0 (mora, foana): coverage + pattern.
    Tier 1 (gating): ghost mate-in-1 raha vitsy ny escape squares fotsiny.
    Voatahiry ao anaty cache, ka tsy averina kajiana ny toerana mitovy."""
    key = (_pos_key(board), color)
    t = _THREAT_CACHE.get(key)
    if t is not None:
        return t
    eks = board.king(not color)
    if eks is None or board.king(color) is None:
        t = Threat(0, 9, 0, "", 0)
    else:
        g = ghost_board(board, color)
        zone = list(chess.SquareSet(chess.BB_KING_ATTACKS[eks]))
        ghost_cov = sum(1 for s in zone if g.is_attacked_by(color, s))
        real_cov = sum(1 for s in zone if board.is_attacked_by(color, s))
        # piece-n'ny king mpanohitra ihany manakana ny escape square (back-rank...)
        blocked = sum(1 for s in zone if board.color_at(s) == (not color)
                      and not g.is_attacked_by(color, s))
        free = len(zone) - ghost_cov - blocked
        cost = ghost_cov - real_cov      # efa mila sacrifice/pressure hanesorana defense
        H0 = defense_potential(board, color, eks) if cost > 0 else 0.0
        score = real_cov * 12 + int(cost * 6 / (1 + 0.25 * H0))   # defense tsy hita
        pb, name = pattern_info(board, color, eks, g)
        score += pb
        tempo, mate = 9, 0
        if g.is_check():
            score += 15
        if g.is_checkmate():
            score += 150
            tempo, mate = 1, 1
        elif free <= 3:                                   # gating
            if ghost_mate_in_1(board, color):
                score += 40
                tempo, mate = 1, 1
            else:
                semi = ghost_mate_in_1(board, color, blockers=True) if blocked else []
                if semi and _real_mate_in_1(board, color, semi):
                    score += 120
                    tempo, mate = 1, 1                # mate tena izy (back-rank, smothered)
                elif semi:
                    score += 25
                    tempo, mate = 2, 2                # cas exceptional
                else:
                    tempo = 2 if free <= 1 else 3
        else:
            tempo = 5 if free <= 5 else 9
        tg = future_targets(board, color)
        if tg and tg[0].net > score:                  # king position hafa no tsara kokoa
            score = tg[0].net
            name = ((name + " ") if name else "") + "=> " + chess.square_name(tg[0].square) \
                + ("=" + tg[0].promo if tg[0].promo else "")
            tempo = min(tempo, tg[0].tempo)
        t = Threat(score, tempo, cost, name, mate)
    if len(_THREAT_CACHE) > 400000:
        _THREAT_CACHE.clear()
    _THREAT_CACHE[key] = t
    return t


def ghost_future(board, color, with_name=False):
    t = threat_profile(board, color)
    return (t.score, t.name) if with_name else t.score


Need = namedtuple("Need", "kind square action cost")
_PLAN_CACHE = {}


def plan_needs(board, color):
    """Inona no tsy maintsy tanterahina mba ho tanteraka ny future (ghost)?
    Vokatry ny ghost: ny efa-voafehy ao amin'ny ghost nefa tsy voafehy amin'ny
    tena board = misy blocker eo amin'ny tsipika. Avy eo vao misy ny decision:
      open_file / open_rank / open_diagonal : blocker eo amin'ny tsipika
      SACRIFICE : blocker voaro, lafo kokoa ny piece-ko mamono azy (cost = very)
      PRESSURE  : azo esorina amin'ny capture/trade/pressure
      block_square (COVER): escape square malalaka tsy voafehy
      clear_own : piece-ko no manakana ny tsipika-ko."""
    key = (_pos_key(board), color)
    r = _PLAN_CACHE.get(key)
    if r is not None:
        return r
    needs = []
    eks = board.king(not color)
    if eks is not None and board.king(color) is not None:
        g = ghost_board(board, color)
        zone = list(chess.SquareSet(chess.BB_KING_ATTACKS[eks]))
        seen = set()
        for s in zone:
            real = board.attackers(color, s)
            for a in g.attackers(color, s):
                if a in real:
                    continue
                for b in chess.SquareSet(chess.between(a, s) & board.occupied):
                    if b in seen:
                        continue
                    seen.add(b)
                    p = board.piece_at(b)
                    if p.color == color:
                        needs.append(Need("clear_own", b, "MOVE_OWN", 0))
                        continue
                    if chess.square_file(a) == chess.square_file(s):
                        kind = "open_file"
                    elif chess.square_rank(a) == chess.square_rank(s):
                        kind = "open_rank"
                    else:
                        kind = "open_diagonal"
                    mine_att = [VAL[board.piece_type_at(x)] for x in board.attackers(color, b)]
                    if not mine_att:
                        action, cost = "PRESSURE", 0
                    elif board.is_attacked_by(not color, b) and min(mine_att) > VAL[p.piece_type]:
                        action, cost = "SACRIFICE", min(mine_att) - VAL[p.piece_type]
                    else:
                        action, cost = "PRESSURE", 0
                    needs.append(Need(kind, b, action, cost))
        for s in zone:
            if not g.is_attacked_by(color, s) and board.piece_at(s) is None:
                needs.append(Need("block_square", s, "COVER", 0))
    if len(_PLAN_CACHE) > 200000:
        _PLAN_CACHE.clear()
    _PLAN_CACHE[key] = needs
    return needs


def _committed_attackers(board, color):
    """Piece an'i `color` izay mandray anjara amin'ny attack-ny (ghost)."""
    eks = board.king(not color)
    if eks is None:
        return set()
    g = ghost_board(board, color)
    out = set()
    for s in king_zone(eks):
        out |= set(g.attackers(color, s))
    out.discard(board.king(color))
    return out


def enable_gain(board, me):
    """Raha tanterahin'ny mpanohitra ny attack-ny, ireo piece ireo (blocker ao
    amin'ny plan_needs-ko) dia mihetsika/miala: manamora ny plan-ko.
    Vokatry ny ghost avokoa: tsy misy fitsipika open-file mitokana."""
    blockers = {n.square for n in plan_needs(board, me) if n.kind.startswith("open")}
    if not blockers:
        return 0
    committed = _committed_attackers(board, not me)
    gain = 0
    for sq in blockers & committed:
        gain += 8 if board.piece_type_at(sq) == PAWN else 10
    return min(gain, 30)


def decide(mine, theirs, me_to_move, enable=0, post=None):
    """Fanapahan-kevitra momba ny attack mpanohitra.
    mine   = threat_profile ahy vs king-ny
    theirs = threat_profile-ny vs king-ko
    enable = enable_gain (fanamorana ny plan-ko avy amin'ny attack-ny)
    post   = post_promo_threat-ny (queen vaovao vs king-ko), raha promotion ny threat-ny
    Mamerina (verdict, care_factor, antony, gain)."""
    if theirs.tempo >= 9 or theirs.score < 25:
        return "IGNORE", 0.0, "tsy mitondra inoninona amin'ny king-ko (laisse tomber)", 0
    my_t = mine.tempo - (0.5 if me_to_move else 0.0)
    th_t = theirs.tempo - (0.0 if me_to_move else 0.5)
    if mine.tempo <= 4 and my_t < th_t and mine.score >= theirs.score * 0.8:
        why = "ny ahy haingana sy mahery kokoa"
        if theirs.name.startswith("Passed pawn"):
            why = "king hunt-ko haingana kokoa noho ny promotion-ny (" + theirs.name + ")"
        return "ALLOW", 0.3, why, 0
    if theirs.tempo >= 5:
        return "ALLOW", 0.6, "mbola lavitra ny attack-ny (tempo >= 5): tsy mila miaro avy hatrany", 0
    if post is not None and theirs.name.startswith("Passed pawn"):
        if post_safe(post):
            if my_t <= th_t + 2:
                return ("ALLOW", 0.3,
                        "aorian'ny promotion dia mbola voafehy ny king-ko, ary ny king hunt-ko "
                        "mbola tonga aloha (" + theirs.name + ")", 0)
        else:
            return ("CARE", 1.0,
                    "aorian'ny promotion dia mampidi-doza ny king-ko ny queen vaovao: mitandrina", 0)
    if theirs.cost >= 3 and theirs.tempo >= 3:
        return "ALLOW", 0.5, "mila sacrifice/pressure be izy, mbola manakana ny defense-ko", 0
    if enable >= 8 and theirs.tempo >= 2:
        return ("ALLOW", 0.5,
                f"ny attack-ny manala ny blocker manakana ny plan-ko (+{enable})",
                enable * 0.5)
    return "CARE", 1.0, "mety hisy fiantraikany amin'ny king safety-ko: mitandrina", 0


def ghost_mate_in_2(board, color):
    """Mate in 2 ao amin'ny ghost (lafo: ampiasaina amin'ny analyze fotsiny)."""
    g = ghost_board(board, color)
    if g.is_check():
        return []
    g.turn = color
    found = []
    for m in list(g.legal_moves):
        g.push(m)
        replies = list(g.legal_moves)
        ok = bool(replies) and not g.is_checkmate()
        if ok:
            for r in replies:
                g.push(r)
                if not _mates_in_1(g, color):
                    ok = False
                g.pop()
                if not ok:
                    break
        g.pop()
        if ok:
            found.append(m)
    return found


# ---------------------------------------------------------------- Evaluation
def _pst_calc(piece_type, sq, color):
    f, r = chess.square_file(sq), chess.square_rank(sq)
    if color == BLACK:
        r = 7 - r
    cd = abs(f - 3.5) + abs(r - 3.5)
    if piece_type == PAWN:
        return r * 6
    if piece_type in (KNIGHT, BISHOP):
        return int(14 - cd * 4)
    if piece_type == QUEEN:
        return int(6 - cd)
    return 0


_PST = {(pt, c): [_pst_calc(pt, sq, c) for sq in range(64)]
        for pt in (PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING) for c in (WHITE, BLACK)}


def pst(piece_type, sq, color):
    return _PST[(piece_type, color)][sq]


def positional_terms(board, eg):
    """Isa maivana (White - Black): bishop pair, rook amin'ny file malalaka, pawn
    structure, fivoaran'ny piece sy toeran'ny king (middlegame)."""
    total = 0
    for color, sign in ((WHITE, 1), (BLACK, -1)):
        mine = board.occupied_co[color]
        own_p = board.pawns & mine
        opp_p = board.pawns & board.occupied_co[not color]
        v = 0
        if _pc(board.bishops & mine) >= 2:
            v += 30
        for rsq in chess.SquareSet(board.rooks & mine):
            fm = chess.BB_FILES[chess.square_file(rsq)]
            if not (fm & own_p):
                v += 20 if not (fm & opp_p) else 10
        files = [own_p & chess.BB_FILES[f] for f in range(8)]
        for f in range(8):
            c = _pc(files[f])
            if not c:
                continue
            if c > 1:
                v -= 12 * (c - 1)
            left = files[f - 1] if f > 0 else 0
            right = files[f + 1] if f < 7 else 0
            if not (left or right):
                v -= 10 * c
        if not eg:
            minors = (board.knights | board.bishops) & mine
            v -= 10 * _pc(minors & (chess.BB_RANK_1 if color == WHITE else chess.BB_RANK_8))
            ksq = board.king(color)
            if ksq is not None:
                kr, kf = chess.square_rank(ksq), chess.square_file(ksq)
                rel = kr if color == WHITE else 7 - kr
                if rel == 0:
                    v += 25 if kf in (1, 2, 6) else (-15 if kf in (3, 4) else 0)
                else:
                    v -= 20 * rel
        total += sign * v
    return total


# ---------------------------------------------------------------- Endgame
ENDGAME_NPM = 2800
_PROMO_CACHE = {}


def non_pawn_material(board):
    return sum(VAL[p.piece_type] for p in board.piece_map().values()
               if p.piece_type not in (PAWN, KING))


def is_endgame(board):
    return non_pawn_material(board) <= ENDGAME_NPM


def is_passed(board, sq, color):
    f, r = chess.square_file(sq), chess.square_rank(sq)
    for e in board.pieces(PAWN, not color):
        ef, er = chess.square_file(e), chess.square_rank(e)
        if abs(ef - f) <= 1 and ((er > r) if color == WHITE else (er < r)):
            return False
    return True


def promo_steps(sq, color):
    r = chess.square_rank(sq)
    return 7 - r if color == WHITE else r


def promo_profile(board, color):
    """Promotion race: passed pawn an'i `color` vs king mpanohitra (ghost) ary
    ny piece mpanohitra afaka manakana azy (cost). tempo = isan'ny move
    mandra-pahazoana queen mavitrika."""
    key = (_pos_key(board), color)
    t = _PROMO_CACHE.get(key)
    if t is not None:
        return t
    best = Threat(0, 9, 0, "", 0)
    eks, ksq = board.king(not color), board.king(color)
    if eks is not None and ksq is not None:
        step = 8 if color == WHITE else -8
        for s in board.pieces(PAWN, color):
            if not is_passed(board, s, color):
                continue
            steps = promo_steps(s, color)
            if steps > 5:
                continue
            path = [s + step * i for i in range(1, steps + 1)]
            if any(board.piece_at(p) for p in path):
                continue                                 # voasakana
            promo = path[-1]
            moves = steps if board.turn != color else steps - 1
            out_of_square = dist(eks, promo) - 1 > moves
            supported = dist(ksq, promo) < dist(eks, promo)
            stoppers = set()
            for p in path:
                for a in board.attackers(not color, p):
                    if board.piece_type_at(a) != KING:
                        stoppers.add(a)
            extra = 2 if stoppers else 0
            if out_of_square:
                tempo, score = steps + 1 + extra, (6 - steps) * 15 + 30
            elif supported:
                tempo, score = steps + 3 + extra, (6 - steps) * 10
            else:
                continue
            cand = Threat(score, tempo, len(stoppers),
                          "Passed pawn " + chess.square_name(s), 0)
            if (cand.tempo, -cand.score) < (best.tempo, -best.score):
                best = cand
    if len(_PROMO_CACHE) > 200000:
        _PROMO_CACHE.clear()
    _PROMO_CACHE[key] = best
    return best


def merge_threat(a, b):
    if b.tempo >= 9 and b.score <= 0:
        return a
    if a.tempo >= 9 and a.score <= 0:
        return b
    hi, lo = (a, b) if (a.tempo, -a.score) <= (b.tempo, -b.score) else (b, a)
    return Threat(hi.score + lo.score * 0.25, min(a.tempo, b.tempo),
                  hi.cost, hi.name, hi.mate)


def full_threat(board, color):
    """King hunt (ghost) + promotion race, atambatra ho threat iray."""
    return merge_threat(threat_profile(board, color), promo_profile(board, color))


def post_promo_threat(board, color):
    """Ghost aorian'ny promotion: raha mahomby ny promotion-n'i `color`, ny queen
    vaovao (miaraka amin'ny piece hafa-ny) vs ny king mpanohitra. Mbola velona
    ve ny king (voafehy), na mampidi-doza? Mamerina Threat na None."""
    pp = promo_profile(board, color)
    if not pp.name.startswith("Passed pawn"):
        return None
    sq = chess.parse_square(pp.name.split()[-1])
    promo = chess.square(chess.square_file(sq), 7 if color == WHITE else 0)
    b = board.copy(stack=False)
    b.remove_piece_at(sq)
    b.set_piece_at(promo, chess.Piece(QUEEN, color))
    return threat_profile(b, color)


def post_safe(post):
    """King voafehy aorian'ny promotion?"""
    return post is None or post.tempo >= 4 or post.score < 25


def kbn_corners(board, color):
    """K+B+N vs K irery: zorony 2 manana loko mitovy amin'ny bishop (tsy maintsy
    ampidirina ny king mpanohitra any). None raha tsy KBN."""
    if board.pawns:
        return None
    if len(board.pieces(BISHOP, color)) != 1 or len(board.pieces(KNIGHT, color)) != 1:
        return None
    if board.pieces(ROOK, color) or board.pieces(QUEEN, color):
        return None
    if any(board.pieces(t, not color) for t in (KNIGHT, BISHOP, ROOK, QUEEN)):
        return None
    b_sq = next(iter(board.pieces(BISHOP, color)))
    light = bool(chess.BB_LIGHT_SQUARES & chess.BB_SQUARES[b_sq])
    return (chess.A8, chess.H1) if light else (chess.A1, chess.H8)


def _manh(a, b):
    return (abs(chess.square_file(a) - chess.square_file(b))
            + abs(chess.square_rank(a) - chess.square_rank(b)))


def _centre_dist(sq):
    return abs(chess.square_file(sq) - 3.5) + abs(chess.square_rank(sq) - 3.5)


def endgame_terms(board):
    """Isa endgame (White POV): king mavitrika, passed pawn + king escort,
    rook ao ambadiky ny pawn, mop-up (alefaso any an-jorony ny king), opposition."""
    eg = is_endgame(board)
    npm = {c: sum(VAL[p.piece_type] for p in board.piece_map().values()
                  if p.color == c and p.piece_type not in (PAWN, KING))
           for c in (WHITE, BLACK)}
    total = 0
    for c in (WHITE, BLACK):
        sign = 1 if c == WHITE else -1
        ksq, eks = board.king(c), board.king(not c)
        if ksq is None or eks is None:
            continue
        v = 0
        if eg:
            v += int((7 - _centre_dist(ksq)) * 4)
        for p in board.pieces(PAWN, c):
            if not is_passed(board, p, c):
                continue
            steps = promo_steps(p, c)
            v += max(0, 6 - steps) * 6
            if eg:
                v += max(0, 4 - dist(ksq, p)) * 4          # king-ko manaraka ny pawn
                v -= max(0, 4 - dist(eks, p)) * 3          # king mpanohitra akaiky
            pf, pr = chess.square_file(p), chess.square_rank(p)
            for rk in board.pieces(ROOK, c):
                if chess.square_file(rk) == pf and (
                        chess.square_rank(rk) < pr if c == WHITE else chess.square_rank(rk) > pr):
                    v += 12                                 # rook ao ambadiky ny pawn
        kc = kbn_corners(board, c)
        if kc is not None:                                  # KBN: zorony mety ihany
            d = min(_manh(eks, x) for x in kc)
            flight = sum(1 for z in chess.SquareSet(chess.BB_KING_ATTACKS[eks])
                         if not board.is_attacked_by(c, z))     # escape malalaka
            v += int((14 - d) * 12 + 4 * (14 - _manh(ksq, eks)) + (8 - flight) * 6)
        elif npm[not c] == 0 and npm[c] >= 450:             # mop-up: king hunt
            v += int(4.7 * _centre_dist(eks) + 1.6 * (14 - _manh(ksq, eks)))
        total += sign * v
    if npm[WHITE] == 0 and npm[BLACK] == 0:                 # opposition (K+P)
        wk, bk = board.king(WHITE), board.king(BLACK)
        if wk is not None and bk is not None:
            same_f = chess.square_file(wk) == chess.square_file(bk)
            same_r = chess.square_rank(wk) == chess.square_rank(bk)
            if (same_f and abs(chess.square_rank(wk) - chess.square_rank(bk)) == 2) or \
               (same_r and abs(chess.square_file(wk) - chess.square_file(bk)) == 2):
                total += 10 if board.turn == BLACK else -10  # ny tsy mandeha no manana opposition
    return total


# ---------------------------------------------------------------- Opening (tanjona JORO)
# Tsy book. Ny opening dia ampiharina amin'ny tanjona telo (vokatry ny evaluation, ny search
# no mifidy ny move, anisan'izany ny gambit raha mendrika ny compensation):
#  1) Fivoarana piece tsy mampiaraka ny hafa  (mobility ny minor piece, passive = sazy)
#  2) Fahefana afovoany d/e (pawn + fanaraha-maso) + file/diagonal malalaka ho an'ny piece
#  3) Manakana ny king mpanohitra tsy hicastle: ghost = ny piece-ko manao ahoana raha tsy
#     misy defense amin'ny toerana lalan'ny castling (e/f/g na e/d/c); ny elanelana ghost-real
#     = ilana sacrifice pawn / open file / open diagonal. Raha efa very ny castling right
#     dia bonus lehibe (king mijanona afovoany).
OPENING_ON = True
OPENING_CAP = 100            # fetra (cp) ho an'ny sosona opening
OPEN_NPM = 4800
_CENTER_SQ = [chess.D4, chess.E4, chess.D5, chess.E5]
_CENTER_PAIR = {WHITE: (chess.D4, chess.E4), BLACK: (chess.D5, chess.E5)}
_BACK = {WHITE: chess.BB_RANK_1, BLACK: chess.BB_RANK_8}


def _undeveloped(board, color):
    return _pc((board.knights | board.bishops) & board.occupied_co[color] & _BACK[color])


def is_opening(board, npm=None):
    """Position fotsiny (tsy miankina amin'ny isan'ny move): mbola betsaka ny piece ary
    mbola tsy vita ny fivoarana."""
    if npm is None:
        npm = non_pawn_material(board)
    return npm >= OPEN_NPM and _undeveloped(board, WHITE) + _undeveloped(board, BLACK) >= 2


def opening_side(board, color, parts=None):
    opp = not color
    own = board.occupied_co[color]
    own_p = board.pawns & own
    back = _BACK[color]
    v = 0.0

    def add(name, x):
        nonlocal v
        v += x
        if parts is not None:
            parts[name] = parts.get(name, 0) + x

    # 1) fivoarana tsy mampiaraka
    minors = (board.knights | board.bishops) & own
    add("dev", 10 * _pc(minors & ~back))
    mob = 0
    for sq in chess.SquareSet(minors):
        m = _pc(board.attacks_mask(sq) & ~own)
        mob += m
        if not (back & chess.BB_SQUARES[sq]):
            pt = board.piece_type_at(sq)
            if (pt == BISHOP and m <= 2) or (pt == KNIGHT and chess.square_file(sq) in (0, 7)):
                add("passive", -10)
    add("mobility", 1.5 * min(mob, 24))
    queens = board.queens & own
    home = chess.D1 if color == WHITE else chess.D8
    if queens and not (queens & chess.BB_SQUARES[home]) and _pc(minors & ~back) < 2:
        add("early_queen", -15)
    for f in (0, 7):
        for sq in chess.SquareSet(own_p & chess.BB_FILES[f]):
            rel = chess.square_rank(sq) if color == WHITE else 7 - chess.square_rank(sq)
            if rel >= 2:
                add("edge_pawn", -6 if rel == 2 else -10)       # pawn sisiny aloha loatra
    mk = board.king(color)                                       # castling ho an'ny king-ko
    my_castled = False
    if mk is not None:
        mkf, mkr = chess.square_file(mk), chess.square_rank(mk)
        my_castled = mkr == (0 if color == WHITE else 7) and mkf in (1, 2, 6, 7)
        if my_castled:
            add("castled", 20)
        elif not board.has_castling_rights(color):
            add("lost_castle", -12)
    if not my_castled:                                           # rook mihetsika aloha
        for rsq in chess.SquareSet(board.rooks & own & back):
            if chess.square_file(rsq) not in (0, 7):
                add("early_rook", -6)

    # 2) afovoany
    for sq in _CENTER_SQ:
        if own_p & chess.BB_SQUARES[sq]:
            add("center_pawn", 14)
        add("center_ctrl", 4 * min(len(board.attackers(color, sq)), 3))
    if all(own_p & chess.BB_SQUARES[s] for s in _CENTER_PAIR[color]):
        add("center_duo", 10)

    # 3) manakana ny castling an'ny mpanohitra (ghost)
    ok = board.king(opp)
    if ok is not None:
        back_rank = 0 if opp == WHITE else 7
        kf, kr = chess.square_file(ok), chess.square_rank(ok)
        castled = kr == back_rank and kf in (1, 2, 6, 7)
        if not castled:
            if not board.has_castling_rights(opp):
                add("castle_denied", 40)                      # tsy afaka icastle intsony
                if kf != 4 or kr != back_rank:
                    add("king_walk", 15)
            else:
                g = chess.Board(None)                          # ghost: piece-ko ihany
                for gs, gp in board.piece_map().items():
                    if gp.color == color:
                        g.set_piece_at(gs, gp)
                att_g = _att_union(g)
                for has, files in ((board.has_kingside_castling_rights(opp), (4, 5, 6)),
                                   (board.has_queenside_castling_rights(opp), (4, 3, 2))):
                    if not has:
                        continue
                    sqs = [chess.square(f, back_rank) for f in files]
                    real = sum(1 for q in sqs if board.is_attacked_by(color, q))
                    gh = sum(1 for q in sqs if att_g & chess.BB_SQUARES[q])
                    add("castle_path", 10 * real + 4 * max(gh - real, 0))
                if board.is_check() and board.turn == opp:
                    add("check", 20)
            for f in (kf - 1, kf, kf + 1):                     # file malalaka mankany amin'ny king
                if 0 <= f <= 7:
                    fm = chess.BB_FILES[f]
                    if (fm & (board.rooks | board.queens) & own) and not (fm & own_p):
                        add("file_vs_king", 14)
            weak = chess.F7 if opp == BLACK else chess.F2
            if kf == 4 and board.piece_at(weak) == chess.Piece(PAWN, opp):
                add("weak_f", 5 * min(len(board.attackers(color, weak)), 3))
    return v


def opening_terms(board):
    d = opening_side(board, WHITE) - opening_side(board, BLACK)
    return max(-OPENING_CAP, min(OPENING_CAP, d))


RR_WEIGHT = 0.5             # lanjan'ny ghost ao amin'ny root re-rank
RR_CAP = 60                 # fetra (cp)
RR_MARGIN = 30              # ny move voasafidy dia tsy ho ambany noho ny tsara indrindra -RR_MARGIN


def joro_layer_delta(board, weight=None, cap=None):
    """Sosona ghost (White POV, voafetra) ho an'ny toerana iray: ghost threat (king hunt +
    promotion), decide (IGNORE/ALLOW/CARE), enable_gain. Ampiasaina amin'ny root re-rank
    (4 toerana isaky ny move), tsy isaky ny node."""
    weight = JORO_WEIGHT if weight is None else weight
    cap = JORO_CAP if cap is None else cap
    att = {WHITE: attack_power(board, WHITE), BLACK: attack_power(board, BLACK)}
    T = {c: full_threat(board, c) for c in (WHITE, BLACK)}
    post = {}
    for c in (WHITE, BLACK):
        post[c] = post_promo_threat(board, c) if T[c].name.startswith("Passed pawn") else None
        if post[c] is not None and not post_safe(post[c]):
            T[c] = Threat(T[c].score + post[c].score * 0.5, T[c].tempo,
                          T[c].cost, T[c].name, T[c].mate)
    care, gain = {}, {}
    for c in (WHITE, BLACK):
        o = not c
        en = 0
        if T[o].score >= 25 and T[o].tempo >= 2 and T[c].cost > 0:
            en = enable_gain(board, c)
        _, care[c], _, gain[c] = decide(T[c], T[o], board.turn == c, en, post[o])
    delta = 0.0
    for c, sign in ((WHITE, 1), (BLACK, -1)):
        delta += sign * ((0.5 * att[c] + T[c].score) * care[not c] + gain[c])
    return max(-cap, min(cap, weight * delta))


LAZY = 350
JORO_LAYERS = True          # False = JORO-lite (tsy misy ghost/threat feno)
JORO_WEIGHT = 0.25          # lanjan'ny sosona ghost ao amin'ny evaluation (0 = tsy misy)
JORO_CAP = 80               # fetra (cp) ho an'ny sosona ghost
ALERT = 10
_EVAL_CACHE = {}
_RANKS_W = chess.BB_RANK_6 | chess.BB_RANK_7
_RANKS_B = chess.BB_RANK_3 | chess.BB_RANK_2


def evaluate(board, alpha=-MATE, beta=MATE):
    """Tier 0 (maivana, foana): material + PST + positional + king safety.
    Tier 1 (rehefa misy 'alert': attack mamely king, endgame, pawn efa akaiky
    ny promotion): sosona JORO feno (ghost, target, decide, endgame)."""
    mat_w = mat_b = npm = 0
    pm = board.piece_map()
    for sq, p in pm.items():
        pt = p.piece_type
        val = VAL[pt] + _PST[(pt, p.color)][sq]
        if p.color == WHITE:
            mat_w += val
        else:
            mat_b += val
        if pt != PAWN and pt != KING:
            npm += VAL[pt]
    base = mat_w - mat_b
    if board.turn == BLACK:
        base = -base
    if base + LAZY <= alpha or base - LAZY >= beta:
        return int(base)
    key = _pos_key(board)
    v = _EVAL_CACHE.get(key)
    if v is not None:
        return v
    eg = npm <= ENDGAME_NPM
    aw, ab = attack_power(board, WHITE), attack_power(board, BLACK)
    pw = board.pawns
    advanced = (pw & board.occupied_co[WHITE] & _RANKS_W) or (pw & board.occupied_co[BLACK] & _RANKS_B)
    score = ((mat_w - mat_b) + positional_terms(board, eg)
             + king_safety(board, WHITE) - king_safety(board, BLACK))
    if OPENING_ON and is_opening(board, npm):                  # opening JORO
        score += opening_terms(board)
    att = {WHITE: aw, BLACK: ab}
    heavy = JORO_LAYERS and (aw >= ALERT or ab >= ALERT or advanced or len(pm) <= 6)
    if not heavy:
        score += 0.5 * (aw - ab)                              # tier 0
        if eg:
            score += endgame_terms(board)                     # king mavitrika, passed pawn
    else:                                                      # tier 1: JORO feno
        T = {c: full_threat(board, c) for c in (WHITE, BLACK)}
        post = {}
        for c in (WHITE, BLACK):
            post[c] = post_promo_threat(board, c) if T[c].name.startswith("Passed pawn") else None
            if post[c] is not None and not post_safe(post[c]):
                T[c] = Threat(T[c].score + post[c].score * 0.5, T[c].tempo,
                              T[c].cost, T[c].name, T[c].mate)
        care, gain = {}, {}
        for c in (WHITE, BLACK):
            o = not c
            en = 0
            if T[o].score >= 25 and T[o].tempo >= 2 and T[c].cost > 0:
                en = enable_gain(board, c)
            _, care[c], _, gain[c] = decide(T[c], T[o], board.turn == c, en, post[o])
        delta = 0.0
        for c, sign in ((WHITE, 1), (BLACK, -1)):
            delta += sign * ((0.5 * att[c] + T[c].score) * care[not c] + gain[c])
        score += max(-JORO_CAP, min(JORO_CAP, JORO_WEIGHT * delta))   # voafetra
        score += endgame_terms(board)
    v = int(score if board.turn == WHITE else -score)
    if len(_EVAL_CACHE) > 300000:
        _EVAL_CACHE.clear()
    _EVAL_CACHE[key] = v
    return v


# ---------------------------------------------------------------- KBNK (W-maneuver)
# Ny W-maneuver dia lava loatra (hatramin'ny 33 move) ho an'ny search. JORO dia
# mamaha ny K+B+N vs K tanteraka amin'ny retrograde analysis (python joro.py buildkbn)
# ary mitahiry "vidin'ny fanerena" (plies hatramin'ny mate) ho an'ny toerana rehetra.
# Rehefa mampiasa azy dia tsy mila numpy ny lalao (bytes fotsiny).
DIRS8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
KDELTAS = [(1, 2), (2, 1), (-1, 2), (-2, 1), (1, -2), (2, -1), (-1, -2), (-2, -1)]
DARK_SQ = [s for s in range(64) if ((s & 7) + (s >> 3)) % 2 == 0]
DARK_IDX = [-1] * 64
for _i, _s in enumerate(DARK_SQ):
    DARK_IDX[_s] = _i


def _kbnk_tables():
    import numpy as np
    KN = np.zeros(4096, bool)
    DIAG = np.zeros(4096, bool)
    BET = np.zeros(4096, np.uint64)
    for a in range(64):
        af, ar = a & 7, a >> 3
        for b in range(64):
            if a == b:
                continue
            bf, br = b & 7, b >> 3
            df, dr = bf - af, br - ar
            if sorted((abs(df), abs(dr))) == [1, 2]:
                KN[a * 64 + b] = True
            if abs(df) == abs(dr):
                DIAG[a * 64 + b] = True
                sf = 1 if df > 0 else -1
                sr = 1 if dr > 0 else -1
                m = 0
                for k in range(1, abs(df)):
                    m |= 1 << ((ar + sr * k) * 8 + af + sf * k)
                BET[a * 64 + b] = m
    return KN, DIAG, BET


def build_kbnk(log=print):
    import numpy as np
    """Retrograde analysis K+B+N vs K (bishop maizina, attaquant = white).
    Mamerina array uint8 : W[idx] = plies hatramin'ny mate (white to move), 255 = tsy fantatra."""
    N = 1 << 23
    KN, DIAG, BET = _kbnk_tables()
    DARK = np.array(DARK_SQ, np.int16)
    DIDX = np.array([max(0, x) for x in DARK_IDX], np.int16)
    idx = np.arange(N, dtype=np.int32)
    n = (idx & 63).astype(np.int16)
    bi = ((idx >> 6) & 31).astype(np.int16)
    bk = ((idx >> 11) & 63).astype(np.int16)
    wk = (idx >> 17).astype(np.int16)
    bs = DARK[bi]
    del idx
    wkf, wkr = wk & 7, wk >> 3
    bkf, bkr = bk & 7, bk >> 3
    bf, br = bs & 7, bs >> 3

    def cheb(af, ar, cf, cr):
        return np.maximum(np.abs(af - cf), np.abs(ar - cr))

    def mk(sq):
        return np.left_shift(np.uint64(1), sq.astype(np.uint64))

    valid = ((wk != bk) & (wk != bs) & (wk != n) & (bk != bs) & (bk != n) & (bs != n)
             & (cheb(wkf, wkr, bkf, bkr) > 1))
    blockers = mk(wk) | mk(n)
    incheck = KN[n * 64 + bk] | (DIAG[bs * 64 + bk] & ((BET[bs * 64 + bk] & blockers) == 0))
    incheck &= valid
    Wvalid = valid & ~incheck

    wk32, bk32, bi32, n32 = wk.astype(np.int32), bk.astype(np.int32), bi.astype(np.int32), n.astype(np.int32)

    # ---- black moves
    NONE, DRAW = N, N + 1
    blk = []
    has_move = np.zeros(N, bool)
    base = (wk32 << 17) | (bi32 << 6) | n32
    for dr, df in DIRS8:
        tf, tr = bkf + df, bkr + dr
        onb = (tf >= 0) & (tf < 8) & (tr >= 0) & (tr < 8)
        t = np.where(onb, tr * 8 + tf, 0).astype(np.int16)
        att_wk = cheb(t & 7, t >> 3, wkf, wkr) <= 1
        capb, capn = (t == bs), (t == n)
        att_n = KN[n * 64 + t] & ~capn
        att_b = DIAG[bs * 64 + t] & ((BET[bs * 64 + t] & blockers) == 0) & ~capb
        legal = onb & valid & ~(att_wk | att_n | att_b)
        res = np.full(N, NONE, np.int32)
        norm = legal & ~(capb | capn)
        res[norm] = (base | (t.astype(np.int32) << 11))[norm]
        res[legal & (capb | capn)] = DRAW
        blk.append(res)
        has_move |= legal
        del tf, tr, onb, t, att_wk, capb, capn, att_n, att_b, legal, norm
    log("black moves ready")

    # ---- white moves
    wm = []
    for dr, df in DIRS8:                                   # king
        tf, tr = wkf + df, wkr + dr
        onb = (tf >= 0) & (tf < 8) & (tr >= 0) & (tr < 8)
        t = np.where(onb, tr * 8 + tf, 0).astype(np.int16)
        ok = onb & valid & (t != bs) & (t != n) & (cheb(t & 7, t >> 3, bkf, bkr) > 1)
        res = np.full(N, N, np.int32)
        res[ok] = ((t.astype(np.int32) << 17) | (bk32 << 11) | (bi32 << 6) | n32)[ok]
        wm.append(res)
    for dr, df in [(1, 1), (1, -1), (-1, 1), (-1, -1)]:    # bishop
        alive = valid.copy()
        for k in range(1, 8):
            pf, pr = bf + df * k, br + dr * k
            onb = (pf >= 0) & (pf < 8) & (pr >= 0) & (pr < 8)
            t = np.where(onb, pr * 8 + pf, 0).astype(np.int16)
            alive &= onb & (t != wk) & (t != n) & (t != bk)
            res = np.full(N, N, np.int32)
            res[alive] = ((wk32 << 17) | (bk32 << 11) | (DIDX[t].astype(np.int32) << 6) | n32)[alive]
            wm.append(res)
    for dr, df in KDELTAS:                                 # knight
        tf, tr = (n & 7) + df, (n >> 3) + dr
        onb = (tf >= 0) & (tf < 8) & (tr >= 0) & (tr < 8)
        t = np.where(onb, tr * 8 + tf, 0).astype(np.int16)
        ok = onb & valid & (t != wk) & (t != bs) & (t != bk)
        res = np.full(N, N, np.int32)
        res[ok] = ((wk32 << 17) | (bk32 << 11) | (bi32 << 6) | t.astype(np.int32))[ok]
        wm.append(res)
    log("white moves ready:", len(wm))

    Bv = np.full(N + 2, 255, np.uint8)
    Wv = np.full(N + 2, 255, np.uint8)
    Wv[N] = 0                                              # NONE -> identity ho an'ny max
    Bv[:N][valid & incheck & ~has_move] = 0                # checkmate
    log("mates:", int((Bv[:N] == 0).sum()))
    it = 0
    while True:
        it += 1
        cand = np.full(N, 255, np.uint8)
        for arr in wm:
            np.minimum(cand, Bv[arr], out=cand)
        newW = Wvalid & (Wv[:N] == 255) & (cand != 255)
        cw = int(newW.sum())
        Wv[:N][newW] = cand[newW] + 1
        m = np.zeros(N, np.uint8)
        for arr in blk:
            np.maximum(m, Wv[arr], out=m)
        newB = valid & (Bv[:N] == 255) & has_move & (m != 255)
        cb = int(newB.sum())
        Bv[:N][newB] = m[newB] + 1
        log(f"iter {it}: W+{cw} B+{cb}")
        if cw == 0 and cb == 0:
            break
    return Wv[:N].copy(), Bv[:N].copy()


KBNK_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kbnk.dat")
KBNK_INF = 10 ** 6


def kbnk_index(board):
    """(attaquant, idx) raha K+B+N vs K irery (tsy misy pawn), na None.
    Normalisation: attaquant = White (flip rank), bishop maizina (flip file)."""
    if board.pawns or len(board.piece_map()) != 4:
        return None
    bs, ns = list(chess.SquareSet(board.bishops)), list(chess.SquareSet(board.knights))
    if len(bs) != 1 or len(ns) != 1:
        return None
    att = board.color_at(bs[0])
    if board.color_at(ns[0]) != att:
        return None

    def tr(sq, flip):
        f, r = chess.square_file(sq), chess.square_rank(sq)
        if att == BLACK:
            r = 7 - r
        if flip:
            f = 7 - f
        return r * 8 + f

    bt = tr(bs[0], False)
    flip = ((bt & 7) + (bt >> 3)) % 2 == 1
    idx = ((tr(board.king(att), flip) * 64 + tr(board.king(not att), flip)) * 32
           + DARK_IDX[tr(bs[0], flip)]) * 64 + tr(ns[0], flip)
    return att, idx


class KBNK:
    def __init__(self, data):
        self.W = data

    @staticmethod
    def load(path=None):
        try:
            with open(path or KBNK_PATH, "rb") as f:
                data = zlib.decompress(f.read())
        except (OSError, zlib.error):
            return None
        return KBNK(data) if len(data) == 1 << 23 else None

    def dtm(self, board):
        """Plies hatramin'ny mate (attaquant mandeha), na None."""
        r = kbnk_index(board)
        if r is None or board.turn != r[0]:
            return None
        v = self.W[r[1]]
        return None if v == 255 else v

    def _defender_value(self, board):
        if board.is_checkmate():
            return 0
        if board.is_game_over():
            return KBNK_INF                      # stalemate / draw
        worst = 0
        for r in list(board.legal_moves):
            board.push(r)
            ix = kbnk_index(board)
            v = 255 if ix is None else self.W[ix[1]]   # None = very piece = draw
            board.pop()
            if v == 255:
                return KBNK_INF
            worst = max(worst, v)
        return worst + 1

    def best_move(self, board):
        """(move, plies hatramin'ny mate) ho an'ny attaquant, na None."""
        if self.dtm(board) is None:
            return None
        best = None
        for m in list(board.legal_moves):
            board.push(m)
            v = self._defender_value(board)
            board.pop()
            if best is None or v < best[0]:
                best = (v, m)
        if best is None or best[0] >= KBNK_INF:
            return None
        return best[1], best[0] + 1


# ---------------------------------------------------------------- Opening book
BOOK_LINES = [
    "e4 e5 Nf3 Nc6 Bb5 a6 Ba4 Nf6 O-O Be7 Re1 b5 Bb3 d6 c3 O-O h3 Nb8 d4 Nbd7",
    "e4 e5 Nf3 Nc6 Bb5 Nf6 O-O Nxe4 d4 Nd6 Bxc6 dxc6 dxe5 Nf5 Qxd8+ Kxd8",
    "e4 e5 Nf3 Nc6 Bc4 Bc5 c3 Nf6 d3 d6 O-O O-O",
    "e4 e5 Nf3 Nc6 Bc4 Nf6 Ng5 d5 exd5 Na5 Bb5+ c6 dxc6 bxc6 Be2 h6 Nf3 e4 Ne5 Bd6",
    "e4 e5 Nf3 Nc6 d4 exd4 Nxd4 Nf6 Nxc6 bxc6 e5 Qe7 Qe2 Nd5 c4 Ba6",
    "e4 e5 Nf3 Nf6 Nxe5 d6 Nf3 Nxe4 d4 d5 Bd3 Nc6 O-O Be7 c4 Nb4 Be2 O-O Nc3 Bf5",
    "e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 a6 Be3 e5 Nb3 Be6 f3 Be7 Qd2 O-O O-O-O Nbd7",
    "e4 c5 Nf3 Nc6 d4 cxd4 Nxd4 g6 Nc3 Bg7 Be3 Nf6 Bc4 O-O Bb3 d6 f3 Bd7 Qd2 Rc8",
    "e4 c5 Nf3 e6 d4 cxd4 Nxd4 Nc6 Nc3 Qc7 Be3 a6 Qd2 Nf6 O-O-O Bb4 f3 Ne5",
    "e4 c5 c3 d5 exd5 Qxd5 d4 Nf6 Nf3 e6 Be2 Nc6 O-O cxd4 cxd4 Be7",
    "e4 e6 d4 d5 Nc3 Nf6 Bg5 Be7 e5 Nfd7 Bxe7 Qxe7 f4 O-O Nf3 c5",
    "e4 e6 d4 d5 Nc3 Bb4 e5 c5 a3 Bxc3+ bxc3 Ne7 Qg4 O-O Bd3 Nbc6",
    "e4 c6 d4 d5 Nc3 dxe4 Nxe4 Bf5 Ng3 Bg6 h4 h6 Nf3 Nd7 h5 Bh7 Bd3 Bxd3 Qxd3 e6",
    "e4 c6 d4 d5 e5 Bf5 Nf3 e6 Be2 c5 Be3 Nd7 O-O Ne7",
    "e4 d5 exd5 Qxd5 Nc3 Qa5 d4 Nf6 Nf3 c6 Bc4 Bf5",
    "e4 d6 d4 Nf6 Nc3 g6 f4 Bg7 Nf3 O-O Bd3 Na6",
    "e4 g6 d4 Bg7 Nc3 d6 Nf3 Nf6 Be2 O-O O-O c6",
    "d4 d5 c4 e6 Nc3 Nf6 Bg5 Be7 e3 O-O Nf3 h6 Bh4 b6",
    "d4 d5 c4 c6 Nf3 Nf6 Nc3 dxc4 a4 Bf5 e3 e6 Bxc4 Bb4 O-O Nbd7",
    "d4 d5 c4 dxc4 Nf3 Nf6 e3 e6 Bxc4 c5 O-O a6",
    "d4 d5 Nf3 Nf6 Bf4 e6 e3 c5 c3 Nc6 Nbd2 Bd6 Bg3 O-O",
    "d4 Nf6 c4 e6 Nc3 Bb4 e3 O-O Bd3 d5 Nf3 c5 O-O Nc6 a3 Bxc3 bxc3 dxc4 Bxc4 Qc7",
    "d4 Nf6 c4 g6 Nc3 Bg7 e4 d6 Nf3 O-O Be2 e5 O-O Nc6 d5 Ne7",
    "d4 Nf6 c4 g6 Nc3 d5 cxd5 Nxd5 e4 Nxc3 bxc3 Bg7 Nf3 c5 Be3 Qa5 Qd2 O-O",
    "d4 Nf6 c4 e6 Nf3 b6 g3 Ba6 b3 Bb4+ Bd2 Be7 Bg2 c6 O-O d5",
    "c4 e5 Nc3 Nf6 Nf3 Nc6 g3 d5 cxd5 Nxd5 Bg2 Nb6 O-O Be7",
    "c4 c5 Nf3 Nf6 Nc3 e6 g3 b6 Bg2 Bb7 O-O Be7 d4 cxd4 Qxd4 d6",
    "Nf3 d5 g3 Nf6 Bg2 e6 O-O Be7 d3 O-O Nbd2 c5 e4 Nc6",
]


def build_book(lines):
    table = {}
    for line in lines:
        b = chess.Board()
        try:
            for san in line.split():
                m = b.parse_san(san)
                entry = table.setdefault(b.epd(), {})
                entry[m.uci()] = entry.get(m.uci(), 0) + 1
                b.push(m)
        except ValueError:
            continue          # line diso: tsy raharahiana ny sisa
    return table


class Book:
    def __init__(self, polyglot_path=None, max_ply=20):
        self.table = build_book(BOOK_LINES)
        self.path = polyglot_path
        self.max_ply = max_ply

    def pick(self, board):
        if board.ply() > self.max_ply:
            return None
        if self.path:
            try:
                with chess.polyglot.open_reader(self.path) as r:
                    return r.weighted_choice(board).move
            except (IndexError, OSError):
                pass
        entry = self.table.get(board.epd())
        if not entry:
            return None
        moves = [chess.Move.from_uci(u) for u in entry]
        weights = list(entry.values())
        legal = [(m, w) for m, w in zip(moves, weights) if m in board.legal_moves]
        if not legal:
            return None
        return random.choices([m for m, _ in legal], [w for _, w in legal])[0]


# ---------------------------------------------------------------- Search
class Timeout(Exception):
    pass


def _tt_store_score(score, ply):
    if score > MATE - 100:
        return score + ply
    if score < -MATE + 100:
        return score - ply
    return score


def _tt_load_score(score, ply):
    if score > MATE - 100:
        return score - ply
    if score < -MATE + 100:
        return score + ply
    return score


class Joro:
    def __init__(self, max_depth=10, time_limit=5.0, book="default", kbnk="default", layers=True,
                 opening_on=True, root_layer=False):
        self.layers = layers
        self.opening_on = opening_on
        self.root_layer = root_layer
        self.kbnk = KBNK.load() if kbnk == "default" else kbnk
        self.max_depth = max_depth
        self.time_limit = time_limit
        self.book = Book() if book == "default" else book
        self.nodes = 0
        self.info = ""
        self.deadline = 0.0
        self.tt = {}
        self.killers = [[None, None] for _ in range(80)]
        self.history = {}
        self.stat_nodes = 0
        self.stat_time = 0.0
        self.stat_moves = 0
        self.stat_depth = 0

    def _check(self):
        self.nodes += 1
        if self.nodes % 128 == 0 and time.time() > self.deadline:
            raise Timeout

    @staticmethod
    def _ttkey(board):
        return (_pos_key(board), board.castling_rights, board.ep_square)

    def order(self, board, moves, ply=0, tt_move=None, check_bonus=True):
        kl = self.killers[min(ply, 79)]
        hist = self.history

        def key(m):
            if m == tt_move:
                return -10 ** 7
            if board.is_capture(m):
                victim = board.piece_type_at(m.to_square) or PAWN
                s = 100000 + 10 * VAL[victim] - VAL[board.piece_type_at(m.from_square)]
            elif m.promotion:
                s = 90000 + VAL[m.promotion]
            else:
                if m == kl[0]:
                    s = 80000
                elif m == kl[1]:
                    s = 79000
                else:
                    s = hist.get((m.from_square, m.to_square), 0)
                if check_bonus and board.gives_check(m):
                    s += 500
            return -s
        return sorted(moves, key=key)

    def quiesce(self, board, alpha, beta, ply):
        self._check()
        stand = evaluate(board, alpha, beta)
        if stand >= beta:
            return beta
        if stand > alpha:
            alpha = stand
        caps = [m for m in board.legal_moves if board.is_capture(m) or m.promotion]
        for m in self.order(board, caps, ply, None, False):
            if not m.promotion:
                victim = board.piece_type_at(m.to_square) or PAWN
                if stand + VAL[victim] + 200 < alpha:        # delta pruning
                    continue
            board.push(m)
            v = -self.quiesce(board, -beta, -alpha, ply + 1)
            board.pop()
            if v >= beta:
                return beta
            if v > alpha:
                alpha = v
        return alpha

    def negamax(self, board, depth, alpha, beta, ply, allow_null=True):
        self._check()
        if board.is_insufficient_material() or board.halfmove_clock >= 100:
            return 0
        if board.halfmove_clock >= 4 and board.is_repetition(2):
            return 0
        in_check = board.is_check()
        if in_check and ply < 12:                              # check extension
            depth += 1
        if depth <= 0:
            return self.quiesce(board, alpha, beta, ply)
        alpha0 = alpha
        key = self._ttkey(board)
        ent = self.tt.get(key)
        tt_move = None
        if ent is not None:
            e_depth, e_flag, e_score, tt_move = ent
            if e_depth >= depth:
                sc = _tt_load_score(e_score, ply)
                if e_flag == 0:
                    return sc
                if e_flag == 1 and sc >= beta:
                    return sc
                if e_flag == 2 and sc <= alpha:
                    return sc
        # null move pruning
        if (allow_null and not in_check and depth >= 3 and beta < MATE - 100
                and board.occupied_co[board.turn] & ~(board.pawns | board.kings)):
            board.push(chess.Move.null())
            v = -self.negamax(board, depth - 3, -beta, -beta + 1, ply + 1, False)
            board.pop()
            if v >= beta:
                return beta
        moves = list(board.legal_moves)
        if not moves:
            return -MATE + ply if in_check else 0
        best_move = None
        best_val = -MATE - 10
        for i, m in enumerate(self.order(board, moves, ply, tt_move)):
            quiet = not board.is_capture(m) and not m.promotion
            board.push(m)
            if i >= 3 and depth >= 3 and quiet and not in_check and not board.is_check():
                red = 2 if (i >= 8 and depth >= 5) else 1           # late move reduction
                v = -self.negamax(board, depth - 1 - red, -alpha - 1, -alpha, ply + 1)
                if v > alpha:
                    v = -self.negamax(board, depth - 1, -beta, -alpha, ply + 1)
            else:
                v = -self.negamax(board, depth - 1, -beta, -alpha, ply + 1)
            board.pop()
            if v > best_val:
                best_val, best_move = v, m
            if v >= beta:
                if quiet:
                    kl = self.killers[min(ply, 79)]
                    if kl[0] != m:
                        kl[1], kl[0] = kl[0], m
                    hk = (m.from_square, m.to_square)
                    self.history[hk] = self.history.get(hk, 0) + depth * depth
                self.tt[key] = (depth, 1, _tt_store_score(beta, ply), m)
                return beta
            if v > alpha:
                alpha = v
        flag = 0 if alpha > alpha0 else 2
        self.tt[key] = (depth, flag, _tt_store_score(alpha, ply), best_move)
        return alpha

    def _root_rerank(self, board, moves, best, best_score, done, t0, n0):
        """JORO = mpandinika ny root: ny search maivana (lite) no mitady, ary ny ghost no mandinika
        ny 4 move voalohany (mitovy depth), ary mifidy ao anatin'ny RR_MARGIN."""
        d = max(1, done - 1)
        cands = [best] + [m for m in moves if m != best][:3]
        self.deadline = t0 + self.time_limit                    # fotoana fanampiny
        sign = 1 if board.turn == WHITE else -1
        res = {}
        try:
            for m in cands:
                board.push(m)
                bonus = sign * joro_layer_delta(board, RR_WEIGHT, RR_CAP)
                v = -self.negamax(board, d - 1, -MATE - 1, MATE + 1, 1)
                board.pop()
                res[m] = (v, bonus)
        except Timeout:
            while len(board.move_stack) > n0:
                board.pop()
        if best not in res or len(res) < 2:
            return best, ""
        top = max(v for v, _ in res.values())
        elig = [m for m, (v, _) in res.items() if v >= top - RR_MARGIN]
        pick = max(elig, key=lambda m: res[m][0] + res[m][1])
        if pick == best:
            return best, f" | JORO root: {board.san(best)} nohazonina"
        return pick, (f" | JORO root: {board.san(pick)} ({res[pick][0]:+d}{res[pick][1]:+.0f}) > "
                      f"{board.san(best)} ({res[best][0]:+d}{res[best][1]:+.0f})")

    def best_move(self, board):
        global JORO_LAYERS, OPENING_ON
        if JORO_LAYERS != self.layers or OPENING_ON != self.opening_on:
            JORO_LAYERS = self.layers                     # JORO feno / JORO-lite
            OPENING_ON = self.opening_on                  # opening JORO / tsy misy
            _EVAL_CACHE.clear()
        if self.book:
            bm = self.book.pick(board)
            if bm is not None:
                self.info = "book"
                return bm
        if self.kbnk is not None:                         # W-maneuver (K+B+N vs K)
            r = self.kbnk.best_move(board)
            if r is not None:
                self.info = f"KBNK table: mate in {(r[1] + 1) // 2}"
                return r[0]
        t0 = time.time()
        self.deadline = t0 + self.time_limit * (0.7 if self.root_layer else 1.0)
        self.nodes = 0
        if len(self.tt) > 600000:
            self.tt.clear()
        self.killers = [[None, None] for _ in range(80)]
        self.history = {}
        n0 = len(board.move_stack)
        moves = self.order(board, list(board.legal_moves))
        fset = ({f.move for f in forcing_moves(board, board.turn)}
                if self.layers else set())                          # forcing moves aloha
        moves.sort(key=lambda m: 0 if m in fset else 1)
        best, best_score = moves[0], 0
        npc = len(board.piece_map())
        extra = 3 if npc <= 4 else 0
        done = 0
        for depth in range(1, self.max_depth + extra + 1):
            alpha, cur_best, cur_score = -MATE - 1, None, None
            try:
                for m in moves:
                    board.push(m)
                    v = -self.negamax(board, depth - 1, -MATE - 1, -alpha, 1)
                    board.pop()
                    if cur_best is None or v > alpha:
                        alpha, cur_best, cur_score = max(v, alpha), m, v
                timed_out = False
                done = depth
            except Timeout:
                while len(board.move_stack) > n0:              # averina ny board
                    board.pop()
                timed_out = True
            if cur_best is not None:                           # misy vokatra (na tsy vita aza)
                best, best_score = cur_best, cur_score
                moves.remove(best)
                moves.insert(0, best)
                el = max(time.time() - t0, 1e-3)
                self.info = (f"depth {depth}{'*' if timed_out else ''}, score {best_score}, "
                             f"nodes {self.nodes}, {int(self.nodes / el)} nps")
            if timed_out or abs(best_score) > MATE - 100:
                break
        if self.root_layer and len(moves) > 1 and abs(best_score) < MATE - 100 and done >= 2:
            best, note = self._root_rerank(board, moves, best, best_score, done, t0, n0)
            self.info += note
        self.stat_nodes += self.nodes
        self.stat_time += time.time() - t0
        self.stat_moves += 1
        self.stat_depth += done
        return best


# ---------------------------------------------------------------- Explain
def explain(board):
    me, opp = board.turn, not board.turn
    names = {WHITE: "White", BLACK: "Black"}
    ks_me, ks_op = king_safety(board, me), king_safety(board, opp)
    Tm, To = full_threat(board, me), full_threat(board, opp)
    en = enable_gain(board, me)
    verdict, factor, why, gain = decide(Tm, To, True, en, post_promo_threat(board, opp))
    print(f"[1] King safety  {names[me]}: {ks_me}  |  {names[opp]}: {ks_op}")
    print("    Is my king safe?  ", "eny" if ks_me >= 10 else "MITANDREMA")
    print("    Is his king safe? ", "eny" if ks_op >= 10 else "TSIA (mety azo atao)")
    print(f"[2] Attack  ahy: {attack_power(board, me)}  |  mpanohitra: {attack_power(board, opp)}")
    print(f"[3] Future ahy (ghost: piece-ko vs king-ny)")
    print(f"    isa {Tm.score} | pattern [{Tm.name or '-'}] | tempo {Tm.tempo} | vidin'ny defense {Tm.cost}")
    pure = ghost_mate_in_1(board, me)
    semi = ghost_mate_in_1(board, me, blockers=True)
    if pure:
        print("    Ghost mate-in-1:", ", ".join(str(m) for m in pure[:5]))
    elif semi:
        real = _real_mate_in_1(board, me, semi)
        print("    " + ("MATE tena izy (back-rank/smothered):" if real
                        else "Cas exceptional (miaraka amin'ny piece mpanohitra):"),
              ", ".join(str(m) for m in semi[:5]))
    elif board.king(opp) is not None and len(board.piece_map()) <= 8:
        m2 = ghost_mate_in_2(board, me)
        if m2:
            print("    Ghost mate-in-2 (nandeha voalohany):", ", ".join(str(m) for m in m2[:5]))
    tg = future_targets(board, me)
    if tg:
        print("[3a] King mpanohitra: toerana mety (tsy ny ankehitriny ihany):")
        forced = forced_destinations(board, me)
        seen_sq, shown = set(), []
        for t in tg:                                   # iray isaky ny toerana
            if t.square not in seen_sq:
                seen_sq.add(t.square)
                shown.append(t)
        for t in shown[:3]:
            tag = "ankehitriny" if t.dist == 0 else f"{t.dist} dingana"
            promo = f", smart promote={t.promo}" if t.promo else ""
            mate = "  [MATE raha tsy misy defense]" if t.mate0 else ""
            print(f"    - king @{chess.square_name(t.square)} ({tag}): net {t.net:.0f}, "
                  f"escape malalaka {t.free}, defense mety {t.hidden:.1f}, "
                  f"tempo {t.tempo}{promo}{mate}")
            for san, dests, only in forced:
                if t.square in dests and t.dist > 0:
                    print(f"        . by force: {san} -> king mankany @{chess.square_name(t.square)}"
                          + (" (tsy misy valiny hafa)" if only else " (mety)"))
            if t.by:
                print(f"        . vidin'ny fanerena nihena noho ny forcing move: {t.by}")
    fm = forcing_moves(board, me, deep=True)
    if fm:
        print("[3b] FORCING MOVES (manery ny king / manala defense / mate threat):")
        for f in fm[:8]:
            extra = f" very {f.cost}" if f.cost else ""
            dd = ""
            if f.dests:
                dd = (" -> king: " + ",".join(chess.square_name(x) for x in f.dests)
                      + (" (tsy misy valiny hafa)" if f.only else ""))
            print(f"    - {f.san} [{f.kind}{extra}]{dd}")
    needs = plan_needs(board, me)
    if needs:
        print("    Zavatra ilaina hanatanterahana ny future (vokatry ny ghost):")
        for n in needs:
            extra = f" (very {n.cost})" if n.cost else ""
            print(f"      - {n.kind} {chess.square_name(n.square)}: {n.action}{extra}")
    print(f"[4] DEFENSE (ghost: piece mpanohitra vs king-ko)")
    print(f"    isa {To.score} | pattern [{To.name or '-'}] | tempo {To.tempo} | vidin'ny defense {To.cost}")
    if To.mate == 1:
        print("    ALOHA: manana ghost mate-in-1 ny mpanohitra raha tsy voasakana!")
    elif To.mate == 2:
        print("    Misy cas exceptional (smothered/blocked) mitahiry ny mpanohitra.")
    if To.cost:
        print(f"    Mila sacrifice/pressure amin'ny {To.cost} escape square izy vao tanteraka.")
    print(f"    Fanapahan-kevitra: {verdict} (lanjan'ny attack-ny x{factor}): {why}")
    if en:
        print(f"    Enabler: ny attack-ny manala blocker ao amin'ny plan-ko (+{en})")
    print("    Can I allow it?   ", "eny" if verdict != "CARE" else "TSIA")
    print("    Do I care?        ", {"IGNORE": "tsia (laisse tomber)", "ALLOW": "kely", "CARE": "eny"}[verdict])
    if is_endgame(board) or any(is_passed(board, p, c)
                                for c in (WHITE, BLACK) for p in board.pieces(PAWN, c)):
        print(f"[5] ENDGAME (material tsy pawn = {non_pawn_material(board)}, "
              f"{'endgame' if is_endgame(board) else 'mbola middlegame'})")
        for c, label in ((me, "ahy"), (opp, "mpanohitra")):
            ps = [p for p in board.pieces(PAWN, c) if is_passed(board, p, c)]
            if ps:
                print(f"    Passed pawn {label}: " + ", ".join(
                    f"{chess.square_name(p)} ({promo_steps(p, c)} dingana)" for p in ps))
            pp = promo_profile(board, c)
            if pp.name:
                print(f"      -> promotion race {label}: tempo {pp.tempo}, "
                      f"mpanakana {pp.cost}, isa {pp.score}")
        print(f"    Race: king hunt ahy tempo {threat_profile(board, me).tempo} "
              f"vs mpanohitra tempo {To.tempo} -> {verdict}")
        for c, label in ((me, "ahy"), (opp, "mpanohitra")):
            pt = post_promo_threat(board, c)
            if pt is not None:
                other = "king-ko" if c == opp else "king-ny"
                print(f"    Aorian'ny promotion {label}: queen vaovao vs {other}: "
                      f"tempo {pt.tempo}, isa {pt.score} -> "
                      f"{(('malemy' if post_safe(pt) else 'MAHERY') if c == me else ('voafehy' if post_safe(pt) else 'MAMPIDI-DOZA'))}")
        for c, label in ((me, "ahy"), (opp, "mpanohitra")):
            kc = kbn_corners(board, c)
            if kc:
                print(f"    KBN {label}: alefaso any amin'ny zorony "
                      + " na ".join(chess.square_name(x) for x in kc))
    if is_opening(board):
        pm, po = {}, {}
        sm, so = opening_side(board, me, pm), opening_side(board, opp, po)
        print(f"[7] OPENING (tanjona JORO): ahy {sm:.0f} | mpanohitra {so:.0f}")
        print("    ahy:", ", ".join(f"{k} {x:+.0f}" for k, x in pm.items()) or "-")
        print("    mpanohitra:", ", ".join(f"{k} {x:+.0f}" for k, x in po.items()) or "-")
    else:
        print("[7] OPENING: vita (miverina amin'ny search + ghost)")
    ki = kbnk_index(board)
    if ki is not None:
        kb = KBNK.load()
        who = "ahy" if ki[0] == me else "mpanohitra"
        if kb is None:
            print("[6] KBNK: tsy hita ny kbnk.dat (alefaso: python joro.py buildkbn)")
        else:
            d = kb.dtm(board)
            r = kb.best_move(board)
            if d is None:
                print(f"[6] KBNK W-maneuver ({who} no attaquant): tsy izy no mandeha")
            else:
                print(f"[6] KBNK W-maneuver ({who}): mate in {(d + 1) // 2} move"
                      + (f", nandeha: {board.san(r[0])}" if r else ""))
    eks = board.king(opp)
    if eks is not None:
        for t in (PAWN, KNIGHT, BISHOP, ROOK, QUEEN):
            for s in board.pieces(t, me):
                print(f"    - {chess.piece_name(t)} {chess.square_name(s)} -> king mpanohitra: "
                      f"{dist(s, eks)} dingana")


class UciPlayer:
    """Engine UCI (oh: joro_cpp) mifanaraka amin'ny interface an'i Joro (best_move, time_limit, tt, stats)."""

    def __init__(self, path, time_limit, options=None):
        import types
        import chess.engine as chess_engine
        self.ce = chess_engine
        self.eng = chess_engine.SimpleEngine.popen_uci(path)
        for k, v in (options or {}).items():
            self.eng.configure({k: v})
        self.time_limit = time_limit
        self.tt = types.SimpleNamespace(clear=lambda: None)
        self.info = ""
        self.stat_nodes = 0
        self.stat_time = 0.0
        self.stat_moves = 0
        self.stat_depth = 0

    def best_move(self, board):
        r = self.eng.play(board, self.ce.Limit(time=self.time_limit), info=self.ce.INFO_ALL)
        inf = r.info or {}
        self.stat_moves += 1
        self.stat_nodes += int(inf.get("nodes", 0) or 0)
        self.stat_time += float(inf.get("time", self.time_limit) or self.time_limit)
        self.stat_depth += int(inf.get("depth", 0) or 0)
        sc = inf.get("score")
        self.info = f"depth {inf.get('depth', '?')}, score {sc.white().score(mate_score=100000) if sc else '?'}, nodes {inf.get('nodes', '?')}"
        return r.move

    def quit(self):
        try:
            self.eng.quit()
        except Exception:                                    # noqa: BLE001
            pass


def run_match(engine, games, sf_path, sf_elo, sf_time, openplies, maxplies, opp="sf", lite=None,
              seed=1000):
    """JORO vs Stockfish (UCI, UCI_Elo voafetra) na JORO feno vs JORO-lite (A/B test).
    Mifandimby loko; ny fiandohana kisendrasendra dia mitovy ho an'ny loko roa."""
    import math
    import shutil
    sf, elo = None, None
    if opp == "sf":
        import chess.engine as chess_engine
        path = sf_path or shutil.which("stockfish") or "/usr/games/stockfish"
        sf = chess_engine.SimpleEngine.popen_uci(path)
        try:
            opt = sf.options.get("UCI_Elo")
            if opt is not None:
                elo = max(opt.min, min(opt.max, sf_elo))
                sf.configure({"UCI_LimitStrength": True, "UCI_Elo": elo})
            else:
                sf.configure({"Skill Level": 3})
        except Exception as e:                               # noqa: BLE001
            print("fampitandremana: tsy azo napetraka ny haavon'ny Stockfish:", e)
        print(f"JORO v{VERSION} vs Stockfish (UCI_Elo={elo}, {sf_time}s/move) "
              f"- {games} lalao, JORO {engine.time_limit}s/move", flush=True)
    else:
        print(f"JORO v{VERSION} FENO vs JORO-LITE - {games} lalao, "
              f"{engine.time_limit}s/move", flush=True)

    def material_adjudicate(b):
        w = sum(VAL[p.piece_type] for p in b.piece_map().values() if p.color == WHITE)
        k = sum(VAL[p.piece_type] for p in b.piece_map().values() if p.color == BLACK)
        cp = w - k
        return cp, ("1-0" if cp >= 400 else ("0-1" if cp <= -400 else "1/2-1/2"))

    w = d = l = 0
    try:
        for g in range(games):
            joro_white = g % 2 == 0
            rng = random.Random(seed + g // 2)
            b = chess.Board()
            for _ in range(openplies):
                legal = list(b.legal_moves)
                if not legal:
                    break
                b.push(rng.choice(legal))
            engine.tt.clear()
            if lite is not None:
                lite.tt.clear()
            while not b.is_game_over(claim_draw=True) and b.ply() < maxplies:
                if (b.turn == WHITE) == joro_white:
                    m = engine.best_move(b)
                elif opp == "lite":
                    m = lite.best_move(b)
                else:
                    m = sf.play(b, chess_engine.Limit(time=sf_time)).move
                b.push(m)
            if b.is_game_over(claim_draw=True):
                res, why = b.result(claim_draw=True), "natao"
            elif sf is not None:                             # adjudication amin'ny Stockfish
                info = sf.analyse(b, chess_engine.Limit(depth=12))
                cp = info["score"].white().score(mate_score=100000)
                res = "1-0" if cp >= 300 else ("0-1" if cp <= -300 else "1/2-1/2")
                why = f"adjudicated ({cp:+d} cp)"
            else:
                cp, res = material_adjudicate(b)
                why = f"adjudicated (material {cp:+d})"
            if res == "1/2-1/2":
                d += 1
                out = "draw"
            elif (res == "1-0") == joro_white:
                w += 1
                out = "JORO nandresy" if opp == "sf" else "FENO nandresy"
            else:
                l += 1
                out = "JORO resy" if opp == "sf" else "LITE nandresy"
            print(f"Lalao {g + 1}: FENO={'White' if joro_white else 'Black'}  {res}  "
                  f"{out}  ({b.ply()} ply, {why})", flush=True)
            print("  ", chess.Board().variation_san(b.move_stack), flush=True)
    finally:
        if sf is not None:
            sf.quit()
    def _st(e):
        if e is None or not e.stat_moves:
            return "-"
        return (f"nps {e.stat_nodes / max(e.stat_time, 1e-3):.0f}, "
                f"depth vita {e.stat_depth / e.stat_moves:.1f}")
    print(f"\nHafainganam-pikarohana: FENO {_st(engine)}"
          + (f" | LITE {_st(lite)}" if lite is not None else ""))
    n = w + d + l
    sc = (w + 0.5 * d) / max(n, 1)
    s2 = min(max(sc, 0.02), 0.98)
    diff = -400 * math.log10(1 / s2 - 1)
    se = math.sqrt(max(sc * (1 - sc), 0.01) / max(n, 1))
    print(f"\nVokatra: +{w} ={d} -{l}  ({sc * 100:.0f}% amin'ny {n} lalao, "
          f"tsy fahazoana +-{se * 100:.0f}%)")
    if opp == "sf" and elo is not None:
        print(f"Elo mombamomba JORO ~ {elo + diff:.0f} (mifototra amin'ny UCI_Elo anaty "
              f"fotoana fohy; tsy azo itokisana raha latsaka ny 30 lalao)")
    elif opp == "lite":
        print(f"JORO feno vs lite: {diff:+.0f} Elo (tsara raha > 0; mila 30+ lalao)")


# ---------------------------------------------------------------- CLI
def play(human_color, engine):
    board = chess.Board()
    while not board.is_game_over():
        print(board, "\n")
        if board.turn == human_color:
            txt = input("Nandeha anao (SAN, oh: e4; 'quit' hijanona): ").strip()
            if txt == "quit":
                return
            try:
                board.push_san(txt)
            except ValueError:
                print("Tsy mety ilay move, avereno.")
        else:
            m = engine.best_move(board)
            print("JORO:", board.san(m), f"({engine.info})")
            board.push(m)
    print(board.result())


def main():
    ap = argparse.ArgumentParser(description="JORO chess engine")
    ap.add_argument("mode", choices=["play", "analyze", "selfplay", "buildkbn", "match"])
    ap.add_argument("arg", nargs="?", default="white", help="white/black na FEN")
    ap.add_argument("--book", default=None, help="Polyglot .bin (tsy voatery)")
    ap.add_argument("--nobook", action="store_true")
    ap.add_argument("--opening", choices=["joro", "book", "none"], default="joro",
                    help="joro = tanjona (default), book = notebook, none")
    ap.add_argument("--lite-opening", choices=["none", "book"], default="none")
    ap.add_argument("--depth", type=int, default=10)
    ap.add_argument("--time", type=float, default=5.0)
    ap.add_argument("--plies", type=int, default=60)
    ap.add_argument("--games", type=int, default=4)
    ap.add_argument("--sf-elo", type=int, default=1500)
    ap.add_argument("--sf-time", type=float, default=0.1)
    ap.add_argument("--sf-path", default=None)
    ap.add_argument("--openplies", type=int, default=2)
    ap.add_argument("--cpp", default=None, help="lalana ny engine C++ (UCI) ho an'ny match/selfplay")
    ap.add_argument("--opp", choices=["sf", "lite"], default="sf")
    ap.add_argument("--seed", type=int, default=1000, help="seed ny fiandohana kisendrasendra")
    ap.add_argument("--joro-weight", type=float, default=None)
    ap.add_argument("--joro-cap", type=float, default=None)
    ap.add_argument("--nolayers", action="store_true", help="JORO-lite ho an'ny engine voalohany")
    ap.add_argument("--joro-mode", choices=["off", "root", "node", "both"], default="root",
                    help="off=lite, root=ghost amin'ny root (default), node=isaky ny node, both")
    ap.add_argument("--rr-weight", type=float, default=None)
    ap.add_argument("--rr-margin", type=float, default=None)
    ap.add_argument("--maxplies", type=int, default=160)
    ap.add_argument("--kbnk", default=None, help="lalana kbnk.dat")
    ap.add_argument("--nokbnk", action="store_true")
    a = ap.parse_args()
    a.arg = a.arg.strip().lstrip(":").strip().strip("'\"").strip()
    if a.arg.lower() in ("startpos", "start"):
        a.arg = chess.STARTING_FEN
    if a.mode == "buildkbn":
        import time as _t
        t0 = _t.time()
        W, _ = build_kbnk(log=lambda *x: print(*x, flush=True))
        out = a.kbnk or KBNK_PATH
        with open(out, "wb") as f:
            f.write(zlib.compress(W.tobytes(), 9))
        print(f"kbnk.dat vita: {out} ({_t.time() - t0:.0f}s, mate indrindra "
              f"{int(W[W != 255].max())} plies)")
        return
    book = Book(a.book) if (a.opening == "book" and not a.nobook) else None
    global JORO_WEIGHT, JORO_CAP, RR_WEIGHT, RR_MARGIN
    if a.joro_weight is not None:
        JORO_WEIGHT = a.joro_weight
    if a.joro_cap is not None:
        JORO_CAP = a.joro_cap
    if a.rr_weight is not None:
        RR_WEIGHT = a.rr_weight
    if a.rr_margin is not None:
        RR_MARGIN = a.rr_margin
    jmode = "off" if a.nolayers else a.joro_mode
    engine = Joro(a.depth, a.time, book, None if a.nokbnk else KBNK.load(a.kbnk),
                  layers=jmode in ("node", "both"), opening_on=(a.opening == "joro"),
                  root_layer=jmode in ("root", "both"))
    if a.cpp:
        engine = UciPlayer(a.cpp, a.time)
    if a.mode == "analyze":
        print(f"JORO v{VERSION}")
        b = chess.Board(a.arg)
        explain(b)
        m = engine.best_move(b)
        print("\nNandeha tsara indrindra:", b.san(m), f"({engine.info})")
    elif a.mode == "match":
        lite = None
        if a.opp == "lite":
            lite = Joro(a.depth, a.time, Book(a.book) if a.lite_opening == "book" else None,
                        None, layers=False, opening_on=False)
        run_match(engine, a.games, a.sf_path, a.sf_elo, a.sf_time, a.openplies, a.maxplies,
                  a.opp, lite, seed=a.seed)
        if a.cpp:
            engine.quit()
    elif a.mode == "selfplay":
        b = chess.Board() if a.arg in ("white", "black") else chess.Board(a.arg)
        print(f"JORO v{VERSION} selfplay (depth {a.depth}, {a.time}s/move, max {a.plies} ply)")
        n = 0
        while not b.is_game_over() and n < a.plies:
            m = engine.best_move(b)
            dots = "." if b.turn == WHITE else "..."
            print(f"{b.fullmove_number}{dots} {b.san(m)}   ({engine.info})", flush=True)
            b.push(m)
            n += 1
        print("Vokatra:", b.result() if b.is_game_over() else "tsy vita (voafetra ny plies)")
        print("FEN farany:", b.fen())
    else:
        play(BLACK if a.arg == "black" else WHITE, engine)


if __name__ == "__main__":
    main()
