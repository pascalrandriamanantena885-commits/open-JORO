// JORO C++ core v0.1  -  bitboard engine (UCI) : move generator, search, evaluation lite.
// Dingana manaraka: ghost (threat, target, forcing), opening, root layer.
// Build: g++ -O3 -std=c++17 -o joro_cpp joro.cpp
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
using namespace std;

typedef uint64_t U64;
typedef int32_t Move;

enum { WHITE = 0, BLACK = 1 };
enum { PAWN = 0, KNIGHT, BISHOP, ROOK, QUEEN, KING };
static const int NOPIECE = 12;
static inline int mk(int c, int pt) { return c * 6 + pt; }
static inline int ptOf(int pc) { return pc % 6; }
static inline int colOf(int pc) { return pc / 6; }
#define BIT(s) (1ULL << (s))
static inline int popcnt(U64 b) { return __builtin_popcountll(b); }
static inline int lsb(U64 b) { return __builtin_ctzll(b); }
static inline int poplsb(U64 &b) { int s = lsb(b); b &= b - 1; return s; }
static inline U64 bswap(U64 x) { return __builtin_bswap64(x); }
static inline int fileOf(int s) { return s & 7; }
static inline int rankOf(int s) { return s >> 3; }

// ---- Move : from(6) to(6) promo(3) flags
static const int F_CAP = 1 << 15, F_EP = 1 << 16, F_CASTLE = 1 << 17, F_DBL = 1 << 18;
static inline int mFrom(Move m) { return m & 63; }
static inline int mTo(Move m) { return (m >> 6) & 63; }
static inline int mPromo(Move m) { return (m >> 12) & 7; }
static inline Move mkMove(int f, int t, int promo = 0, int flags = 0) { return f | (t << 6) | (promo << 12) | flags; }

// ---- tables
static U64 KNIGHT_ATT[64], KING_ATT[64], PAWN_ATT[2][64];
static U64 FILE_SQ[64], DIAG_SQ[64], ANTI_SQ[64];     // tsipika mamakivaky ny sq (tsy ao anatiny)
static U64 FILE_BB[8], RANK_BB[8];
static U64 PASSMASK[2][64];
static uint8_t RANK_ATT[8][256];
static U64 ZP[12][64], ZCAST[16], ZEPF[8], ZSIDE;
static int PSQ[12][64];
static const int VALS[6] = {100, 320, 330, 500, 900, 0};
static const int ATTW[6] = {1, 2, 2, 3, 5, 0};
static int CMASK[64];

static U64 rng_state = 0x9E3779B97F4A7C15ULL;
static U64 rnd64() {
    rng_state ^= rng_state << 13; rng_state ^= rng_state >> 7; rng_state ^= rng_state << 17;
    return rng_state * 0x2545F4914F6CDD1DULL;
}

static int pstCalc(int pt, int sq, int color) {
    int f = fileOf(sq), r = rankOf(sq);
    if (color == BLACK) r = 7 - r;
    double cd = fabs(f - 3.5) + fabs(r - 3.5);
    if (pt == PAWN) return r * 6;
    if (pt == KNIGHT || pt == BISHOP) return (int)(14 - cd * 4);
    if (pt == QUEEN) return (int)(6 - cd);
    return 0;
}

static void initTables() {
    for (int s = 0; s < 64; s++) {
        int f = fileOf(s), r = rankOf(s);
        KNIGHT_ATT[s] = KING_ATT[s] = PAWN_ATT[0][s] = PAWN_ATT[1][s] = 0;
        static const int kn[8][2] = {{1, 2}, {2, 1}, {-1, 2}, {-2, 1}, {1, -2}, {2, -1}, {-1, -2}, {-2, -1}};
        for (auto &d : kn) {
            int nf = f + d[0], nr = r + d[1];
            if (nf >= 0 && nf < 8 && nr >= 0 && nr < 8) KNIGHT_ATT[s] |= BIT(nr * 8 + nf);
        }
        for (int df = -1; df <= 1; df++)
            for (int dr = -1; dr <= 1; dr++) {
                if (!df && !dr) continue;
                int nf = f + df, nr = r + dr;
                if (nf >= 0 && nf < 8 && nr >= 0 && nr < 8) KING_ATT[s] |= BIT(nr * 8 + nf);
            }
        for (int df = -1; df <= 1; df += 2) {
            if (f + df >= 0 && f + df < 8) {
                if (r + 1 < 8) PAWN_ATT[WHITE][s] |= BIT((r + 1) * 8 + f + df);
                if (r - 1 >= 0) PAWN_ATT[BLACK][s] |= BIT((r - 1) * 8 + f + df);
            }
        }
        FILE_SQ[s] = DIAG_SQ[s] = ANTI_SQ[s] = 0;
        for (int i = 0; i < 8; i++) if (i != r) FILE_SQ[s] |= BIT(i * 8 + f);
        for (int i = -7; i <= 7; i++) {
            if (i == 0) continue;
            int nf = f + i, nr = r + i;
            if (nf >= 0 && nf < 8 && nr >= 0 && nr < 8) DIAG_SQ[s] |= BIT(nr * 8 + nf);
            nf = f + i; nr = r - i;
            if (nf >= 0 && nf < 8 && nr >= 0 && nr < 8) ANTI_SQ[s] |= BIT(nr * 8 + nf);
        }
    }
    for (int f = 0; f < 8; f++) {
        FILE_BB[f] = 0; RANK_BB[f] = 0;
        for (int i = 0; i < 8; i++) { FILE_BB[f] |= BIT(i * 8 + f); RANK_BB[f] |= BIT(f * 8 + i); }
    }
    for (int s = 0; s < 64; s++) {
        int f = fileOf(s), r = rankOf(s);
        PASSMASK[WHITE][s] = PASSMASK[BLACK][s] = 0;
        for (int ff = max(0, f - 1); ff <= min(7, f + 1); ff++) {
            for (int rr = r + 1; rr < 8; rr++) PASSMASK[WHITE][s] |= BIT(rr * 8 + ff);
            for (int rr = 0; rr < r; rr++) PASSMASK[BLACK][s] |= BIT(rr * 8 + ff);
        }
    }
    for (int f = 0; f < 8; f++)
        for (int o = 0; o < 256; o++) {
            int a = 0;
            for (int i = f + 1; i < 8; i++) { a |= 1 << i; if (o & (1 << i)) break; }
            for (int i = f - 1; i >= 0; i--) { a |= 1 << i; if (o & (1 << i)) break; }
            RANK_ATT[f][o] = (uint8_t)a;
        }
    for (int p = 0; p < 12; p++) for (int s = 0; s < 64; s++) ZP[p][s] = rnd64();
    for (int i = 0; i < 16; i++) ZCAST[i] = rnd64();
    for (int i = 0; i < 8; i++) ZEPF[i] = rnd64();
    ZSIDE = rnd64();
    for (int p = 0; p < 12; p++)
        for (int s = 0; s < 64; s++) {
            int c = colOf(p), pt = ptOf(p);
            int v = VALS[pt] + pstCalc(pt, s, c);
            PSQ[p][s] = (c == WHITE) ? v : -v;
        }
    for (int s = 0; s < 64; s++) CMASK[s] = 15;
    CMASK[0] = 13; CMASK[4] = 12; CMASK[7] = 14; CMASK[56] = 7; CMASK[60] = 3; CMASK[63] = 11;
}

static inline U64 hq(U64 occ, int sq, U64 mask) {
    U64 s = BIT(sq);
    U64 fwd = occ & mask, rev = bswap(fwd);
    fwd -= s; rev -= bswap(s);
    fwd ^= bswap(rev);
    return fwd & mask;
}
static inline U64 rankAtt(U64 occ, int sq) {
    int r = rankOf(sq), f = fileOf(sq);
    return (U64)RANK_ATT[f][(occ >> (8 * r)) & 0xFF] << (8 * r);
}
static inline U64 rookAtt(U64 occ, int sq) { return hq(occ, sq, FILE_SQ[sq]) | rankAtt(occ, sq); }
static inline U64 bishopAtt(U64 occ, int sq) { return hq(occ, sq, DIAG_SQ[sq]) | hq(occ, sq, ANTI_SQ[sq]); }
static inline U64 attacksFrom(int pt, int sq, U64 occ, int c) {
    switch (pt) {
        case PAWN: return PAWN_ATT[c][sq];
        case KNIGHT: return KNIGHT_ATT[sq];
        case BISHOP: return bishopAtt(occ, sq);
        case ROOK: return rookAtt(occ, sq);
        case QUEEN: return bishopAtt(occ, sq) | rookAtt(occ, sq);
        default: return KING_ATT[sq];
    }
}

// ---- Position
struct Undo { U64 key; Move move; int cap, castling, ep, halfmove, score; };
struct Position {
    U64 bb[12], occ[2], all;
    int8_t sq[64];
    int stm, castling, ep, halfmove, fullmove, score;
    U64 key;
    Undo hist[4096];
    int hply;
};

static inline int kingSq(const Position &p, int c) { return lsb(p.bb[mk(c, KING)]); }
static inline bool isAttacked(const Position &p, int s, int by) {
    if (PAWN_ATT[by ^ 1][s] & p.bb[mk(by, PAWN)]) return true;
    if (KNIGHT_ATT[s] & p.bb[mk(by, KNIGHT)]) return true;
    if (KING_ATT[s] & p.bb[mk(by, KING)]) return true;
    if (bishopAtt(p.all, s) & (p.bb[mk(by, BISHOP)] | p.bb[mk(by, QUEEN)])) return true;
    if (rookAtt(p.all, s) & (p.bb[mk(by, ROOK)] | p.bb[mk(by, QUEEN)])) return true;
    return false;
}
static inline bool inCheck(const Position &p) { return isAttacked(p, kingSq(p, p.stm), p.stm ^ 1); }

static inline void addPc(Position &p, int pc, int s) {
    p.bb[pc] |= BIT(s); p.occ[colOf(pc)] |= BIT(s); p.all |= BIT(s); p.sq[s] = (int8_t)pc;
    p.key ^= ZP[pc][s]; p.score += PSQ[pc][s];
}
static inline void delPc(Position &p, int pc, int s) {
    p.bb[pc] &= ~BIT(s); p.occ[colOf(pc)] &= ~BIT(s); p.all &= ~BIT(s); p.sq[s] = NOPIECE;
    p.key ^= ZP[pc][s]; p.score -= PSQ[pc][s];
}

static void clearPos(Position &p) {
    memset(p.bb, 0, sizeof p.bb); p.occ[0] = p.occ[1] = p.all = 0;
    for (int i = 0; i < 64; i++) p.sq[i] = NOPIECE;
    p.stm = WHITE; p.castling = 0; p.ep = -1; p.halfmove = 0; p.fullmove = 1; p.score = 0; p.key = 0; p.hply = 0;
}

static bool setFen(Position &p, const string &fen) {
    clearPos(p);
    istringstream is(fen);
    string board, side, cast, ep;
    int hm = 0, fm = 1;
    if (!(is >> board >> side >> cast >> ep)) return false;
    is >> hm >> fm;
    int r = 7, f = 0;
    for (char ch : board) {
        if (ch == '/') { r--; f = 0; }
        else if (isdigit((unsigned char)ch)) f += ch - '0';
        else {
            int c = isupper((unsigned char)ch) ? WHITE : BLACK, pt = -1;
            switch (tolower(ch)) { case 'p': pt = PAWN; break; case 'n': pt = KNIGHT; break; case 'b': pt = BISHOP; break;
                case 'r': pt = ROOK; break; case 'q': pt = QUEEN; break; case 'k': pt = KING; break; }
            if (pt < 0 || r < 0 || f > 7) return false;
            addPc(p, mk(c, pt), r * 8 + f); f++;
        }
    }
    p.stm = (side == "w") ? WHITE : BLACK;
    if (p.stm == BLACK) p.key ^= ZSIDE;
    for (char ch : cast) {
        if (ch == 'K') p.castling |= 1; else if (ch == 'Q') p.castling |= 2;
        else if (ch == 'k') p.castling |= 4; else if (ch == 'q') p.castling |= 8;
    }
    p.key ^= ZCAST[p.castling];
    p.ep = -1;
    if (ep != "-" && ep.size() >= 2) { p.ep = (ep[1] - '1') * 8 + (ep[0] - 'a'); p.key ^= ZEPF[fileOf(p.ep)]; }
    p.halfmove = hm; p.fullmove = fm;
    return popcnt(p.bb[mk(WHITE, KING)]) == 1 && popcnt(p.bb[mk(BLACK, KING)]) == 1;
}
static const char *STARTFEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

static string sqName(int s) { string r; r += (char)('a' + fileOf(s)); r += (char)('1' + rankOf(s)); return r; }
static string moveStr(Move m) {
    string s = sqName(mFrom(m)) + sqName(mTo(m));
    static const char pc[] = "pnbrqk";
    if (mPromo(m)) s += pc[mPromo(m)];
    return s;
}

// ---- make / unmake
static bool makeMove(Position &p, Move m) {
    Undo &u = p.hist[p.hply++];
    u.key = p.key; u.move = m; u.castling = p.castling; u.ep = p.ep; u.halfmove = p.halfmove; u.score = p.score;
    int from = mFrom(m), to = mTo(m), c = p.stm, pc = p.sq[from], pt = ptOf(pc);
    int cap = p.sq[to];
    p.key ^= ZSIDE;
    if (p.ep >= 0) p.key ^= ZEPF[fileOf(p.ep)];
    p.ep = -1;
    if (m & F_EP) {
        int cs = to + (c == WHITE ? -8 : 8);
        cap = p.sq[cs];
        delPc(p, cap, cs);
    } else if (cap != NOPIECE) delPc(p, cap, to);
    u.cap = cap;
    delPc(p, pc, from);
    if (mPromo(m)) addPc(p, mk(c, mPromo(m)), to); else addPc(p, pc, to);
    if (m & F_CASTLE) {
        if (to == from + 2) { delPc(p, mk(c, ROOK), from + 3); addPc(p, mk(c, ROOK), from + 1); }
        else { delPc(p, mk(c, ROOK), from - 4); addPc(p, mk(c, ROOK), from - 1); }
    }
    if (m & F_DBL) { p.ep = from + (c == WHITE ? 8 : -8); p.key ^= ZEPF[fileOf(p.ep)]; }
    p.key ^= ZCAST[p.castling];
    p.castling &= CMASK[from] & CMASK[to];
    p.key ^= ZCAST[p.castling];
    p.halfmove = (pt == PAWN || cap != NOPIECE) ? 0 : p.halfmove + 1;
    if (c == BLACK) p.fullmove++;
    p.stm ^= 1;
    if (isAttacked(p, kingSq(p, c), c ^ 1)) {
        // illegal : averina
        Undo &uu = p.hist[--p.hply];
        p.stm ^= 1; if (c == BLACK) p.fullmove--;
        // averina ny piece
        if (m & F_CASTLE) {
            if (to == from + 2) { delPc(p, mk(c, ROOK), from + 1); addPc(p, mk(c, ROOK), from + 3); }
            else { delPc(p, mk(c, ROOK), from - 1); addPc(p, mk(c, ROOK), from - 4); }
        }
        if (mPromo(m)) delPc(p, mk(c, mPromo(m)), to); else delPc(p, pc, to);
        addPc(p, pc, from);
        if (m & F_EP) addPc(p, cap, to + (c == WHITE ? -8 : 8));
        else if (cap != NOPIECE) addPc(p, cap, to);
        p.castling = uu.castling; p.ep = uu.ep; p.halfmove = uu.halfmove; p.key = uu.key; p.score = uu.score;
        return false;
    }
    return true;
}
static void unmakeMove(Position &p) {
    Undo &u = p.hist[--p.hply];
    Move m = u.move;
    p.stm ^= 1;
    int c = p.stm, from = mFrom(m), to = mTo(m);
    if (c == BLACK) p.fullmove--;
    int pc;
    if (mPromo(m)) { delPc(p, mk(c, mPromo(m)), to); pc = mk(c, PAWN); }
    else { pc = p.sq[to]; delPc(p, pc, to); }
    addPc(p, pc, from);
    if (m & F_CASTLE) {
        if (to == from + 2) { delPc(p, mk(c, ROOK), from + 1); addPc(p, mk(c, ROOK), from + 3); }
        else { delPc(p, mk(c, ROOK), from - 1); addPc(p, mk(c, ROOK), from - 4); }
    }
    if (u.cap != NOPIECE) {
        if (m & F_EP) addPc(p, u.cap, to + (c == WHITE ? -8 : 8)); else addPc(p, u.cap, to);
    }
    p.castling = u.castling; p.ep = u.ep; p.halfmove = u.halfmove; p.key = u.key; p.score = u.score;
}
static void makeNull(Position &p) {
    Undo &u = p.hist[p.hply++];
    u.key = p.key; u.move = 0; u.castling = p.castling; u.ep = p.ep; u.halfmove = p.halfmove; u.score = p.score; u.cap = NOPIECE;
    if (p.ep >= 0) p.key ^= ZEPF[fileOf(p.ep)];
    p.ep = -1; p.key ^= ZSIDE; p.stm ^= 1; p.halfmove++;
}
static void unmakeNull(Position &p) {
    Undo &u = p.hist[--p.hply];
    p.stm ^= 1; p.castling = u.castling; p.ep = u.ep; p.halfmove = u.halfmove; p.key = u.key; p.score = u.score;
}

// ---- move generation (pseudo-legal)
struct MoveList { Move m[256]; int sc[256]; int n; };
static inline void addMove(MoveList &l, Move m) { l.m[l.n++] = m; }
static void genMoves(const Position &p, MoveList &l, bool capsOnly) {
    l.n = 0;
    int c = p.stm, o = c ^ 1;
    U64 own = p.occ[c], opp = p.occ[o], all = p.all;
    int dir = c == WHITE ? 8 : -8, promoR = c == WHITE ? 6 : 1, startR = c == WHITE ? 1 : 6;
    static const int PRO[4] = {QUEEN, ROOK, BISHOP, KNIGHT};
    U64 pw = p.bb[mk(c, PAWN)];
    while (pw) {
        int from = poplsb(pw), r = rankOf(from), fwd = from + dir;
        if (!(all & BIT(fwd))) {
            if (r == promoR) {
                for (int i = 0; i < 4; i++) { if (capsOnly && i > 0) break; addMove(l, mkMove(from, fwd, PRO[i])); }
            } else if (!capsOnly) {
                addMove(l, mkMove(from, fwd));
                if (r == startR && !(all & BIT(fwd + dir))) addMove(l, mkMove(from, fwd + dir, 0, F_DBL));
            }
        }
        U64 caps = PAWN_ATT[c][from] & opp;
        while (caps) {
            int to = poplsb(caps);
            if (r == promoR) { for (int i = 0; i < 4; i++) addMove(l, mkMove(from, to, PRO[i], F_CAP)); }
            else addMove(l, mkMove(from, to, 0, F_CAP));
        }
        if (p.ep >= 0 && (PAWN_ATT[c][from] & BIT(p.ep))) addMove(l, mkMove(from, p.ep, 0, F_EP | F_CAP));
    }
    for (int pt = KNIGHT; pt <= KING; pt++) {
        U64 b = p.bb[mk(c, pt)];
        while (b) {
            int from = poplsb(b);
            U64 t = attacksFrom(pt, from, all, c) & ~own;
            if (capsOnly) t &= opp;
            while (t) {
                int to = poplsb(t);
                addMove(l, mkMove(from, to, 0, (opp & BIT(to)) ? F_CAP : 0));
            }
        }
    }
    if (!capsOnly && p.castling) {
        if (c == WHITE) {
            if ((p.castling & 1) && !(all & (BIT(5) | BIT(6))) && !isAttacked(p, 4, BLACK) && !isAttacked(p, 5, BLACK) && !isAttacked(p, 6, BLACK))
                addMove(l, mkMove(4, 6, 0, F_CASTLE));
            if ((p.castling & 2) && !(all & (BIT(1) | BIT(2) | BIT(3))) && !isAttacked(p, 4, BLACK) && !isAttacked(p, 3, BLACK) && !isAttacked(p, 2, BLACK))
                addMove(l, mkMove(4, 2, 0, F_CASTLE));
        } else {
            if ((p.castling & 4) && !(all & (BIT(61) | BIT(62))) && !isAttacked(p, 60, WHITE) && !isAttacked(p, 61, WHITE) && !isAttacked(p, 62, WHITE))
                addMove(l, mkMove(60, 62, 0, F_CASTLE));
            if ((p.castling & 8) && !(all & (BIT(57) | BIT(58) | BIT(59))) && !isAttacked(p, 60, WHITE) && !isAttacked(p, 59, WHITE) && !isAttacked(p, 58, WHITE))
                addMove(l, mkMove(60, 58, 0, F_CASTLE));
        }
    }
}

static U64 perft(Position &p, int depth) {
    if (depth == 0) return 1;
    MoveList l; genMoves(p, l, false);
    U64 n = 0;
    for (int i = 0; i < l.n; i++) {
        if (!makeMove(p, l.m[i])) continue;
        n += (depth == 1) ? 1 : perft(p, depth - 1);
        unmakeMove(p);
    }
    return n;
}

// =====================================================================================
// EVALUATION (lite) - mitovy amin'ny tier-0 ao amin'ny joro.py
// =====================================================================================
static inline int distCheb(int a, int b) { return max(abs(fileOf(a) - fileOf(b)), abs(rankOf(a) - rankOf(b))); }
static inline int distManh(int a, int b) { return abs(fileOf(a) - fileOf(b)) + abs(rankOf(a) - rankOf(b)); }
static inline double centreDist(int s) { return fabs(fileOf(s) - 3.5) + fabs(rankOf(s) - 3.5); }
static inline int floorDiv2(int x) { return x >= 0 ? x / 2 : -((-x + 1) / 2); }

static int npmOf(const Position &p, int c) {
    return popcnt(p.bb[mk(c, KNIGHT)]) * 320 + popcnt(p.bb[mk(c, BISHOP)]) * 330 +
           popcnt(p.bb[mk(c, ROOK)]) * 500 + popcnt(p.bb[mk(c, QUEEN)]) * 900;
}

static int kingSafety(const Position &p, int c) {
    int ksq = kingSq(p, c);
    U64 zone = KING_ATT[ksq];
    int defenders = popcnt(zone & p.occ[c]);
    int step = c == WHITE ? 8 : -8, kf = fileOf(ksq), shield = 0, open = 0;
    U64 ownp = p.bb[mk(c, PAWN)];
    for (int df = -1; df <= 1; df++) {
        int f = kf + df;
        if (f < 0 || f > 7) continue;
        int s = ksq + step + df;
        if (s >= 0 && s < 64 && (ownp & BIT(s))) shield++;
        if (!(ownp & FILE_BB[f])) open++;
    }
    int score = defenders * 6 + shield * 8 - open * 12;
    if (!p.bb[mk(c ^ 1, QUEEN)]) score = floorDiv2(score);
    return score;
}

static int attackPower(const Position &p, int c) {
    int eks = kingSq(p, c ^ 1);
    U64 zone = KING_ATT[eks] | BIT(eks);
    int total = 0, n = 0;
    for (int pt = PAWN; pt <= QUEEN; pt++) {
        U64 b = p.bb[mk(c, pt)];
        while (b) {
            int s = poplsb(b);
            int cnt = popcnt(attacksFrom(pt, s, p.all, c) & zone);
            if (cnt) { total += ATTW[pt] * cnt; n++; }
        }
    }
    return total * min(n, 4);
}

static int positionalOf(const Position &p, bool eg, int c) {
    U64 ownp = p.bb[mk(c, PAWN)], oppp = p.bb[mk(c ^ 1, PAWN)];
    int v = 0;
    if (popcnt(p.bb[mk(c, BISHOP)]) >= 2) v += 30;
    U64 rk = p.bb[mk(c, ROOK)];
    while (rk) {
        int s = poplsb(rk);
        U64 fm = FILE_BB[fileOf(s)];
        if (!(fm & ownp)) v += (fm & oppp) ? 10 : 20;
    }
    int cnt[8];
    for (int f = 0; f < 8; f++) cnt[f] = popcnt(ownp & FILE_BB[f]);
    for (int f = 0; f < 8; f++) {
        if (!cnt[f]) continue;
        if (cnt[f] > 1) v -= 12 * (cnt[f] - 1);
        int l = f > 0 ? cnt[f - 1] : 0, r = f < 7 ? cnt[f + 1] : 0;
        if (!l && !r) v -= 10 * cnt[f];
    }
    if (!eg) {
        U64 minors = p.bb[mk(c, KNIGHT)] | p.bb[mk(c, BISHOP)];
        v -= 10 * popcnt(minors & (c == WHITE ? RANK_BB[0] : RANK_BB[7]));
        int ksq = kingSq(p, c), kr = rankOf(ksq), kf = fileOf(ksq);
        int rel = c == WHITE ? kr : 7 - kr;
        if (rel == 0) v += (kf == 1 || kf == 2 || kf == 6) ? 25 : ((kf == 3 || kf == 4) ? -15 : 0);
        else v -= 20 * rel;
    }
    return v;
}

static int endgameTerms(const Position &p, int npmW, int npmB) {
    int npmv[2] = {npmW, npmB}, total = 0;
    for (int c = 0; c < 2; c++) {
        int sign = c == WHITE ? 1 : -1, ksq = kingSq(p, c), eks = kingSq(p, c ^ 1);
        int v = (int)((7 - centreDist(ksq)) * 4);
        U64 pw = p.bb[mk(c, PAWN)];
        while (pw) {
            int s = poplsb(pw);
            if (PASSMASK[c][s] & p.bb[mk(c ^ 1, PAWN)]) continue;
            int steps = c == WHITE ? 7 - rankOf(s) : rankOf(s);
            v += max(0, 6 - steps) * 6;
            v += max(0, 4 - distCheb(ksq, s)) * 4;
            v -= max(0, 4 - distCheb(eks, s)) * 3;
            U64 rk = p.bb[mk(c, ROOK)] & FILE_BB[fileOf(s)];
            while (rk) {
                int rs = poplsb(rk);
                if (c == WHITE ? rankOf(rs) < rankOf(s) : rankOf(rs) > rankOf(s)) v += 12;
            }
        }
        if (npmv[c ^ 1] == 0 && npmv[c] >= 450) v += (int)(4.7 * centreDist(eks) + 1.6 * (14 - distManh(ksq, eks)));
        total += sign * v;
    }
    if (npmW == 0 && npmB == 0) {
        int wk = kingSq(p, WHITE), bk = kingSq(p, BLACK);
        bool sf = fileOf(wk) == fileOf(bk), sr = rankOf(wk) == rankOf(bk);
        if ((sf && abs(rankOf(wk) - rankOf(bk)) == 2) || (sr && abs(fileOf(wk) - fileOf(bk)) == 2))
            total += (p.stm == BLACK) ? 10 : -10;
    }
    return total;
}


// =====================================================================================
// OPENING (tanjona JORO) - port avy amin'ny joro.py v1.4.1 : fivoarana tsy mampiaraka, afovoany,
// castling ho an'ny king-ko, ary FANAKANANA ny castling an'ny mpanohitra (ghost).
// =====================================================================================
static bool OPENING_ON = false;
static int OPEN_CAP = 100;
static const int OPEN_NPM = 4800;
static U64 BACKRANK[2];
static const int SQ_D1 = 3, SQ_D8 = 59, SQ_F2 = 13, SQ_F7 = 53, SQ_D4 = 27, SQ_E4 = 28, SQ_D5 = 35, SQ_E5 = 36;

static U64 attackersTo(const Position &p, int s, int c) {
    U64 a = (PAWN_ATT[c ^ 1][s] & p.bb[mk(c, PAWN)]) | (KNIGHT_ATT[s] & p.bb[mk(c, KNIGHT)]) | (KING_ATT[s] & p.bb[mk(c, KING)]);
    a |= bishopAtt(p.all, s) & (p.bb[mk(c, BISHOP)] | p.bb[mk(c, QUEEN)]);
    a |= rookAtt(p.all, s) & (p.bb[mk(c, ROOK)] | p.bb[mk(c, QUEEN)]);
    return a;
}
static inline int undeveloped(const Position &p, int c) {
    return popcnt((p.bb[mk(c, KNIGHT)] | p.bb[mk(c, BISHOP)]) & BACKRANK[c]);
}
static bool isOpeningPos(const Position &p, int npmW, int npmB) {
    return npmW + npmB >= OPEN_NPM && undeveloped(p, WHITE) + undeveloped(p, BLACK) >= 2;
}
static double openingSide(const Position &p, int c) {
    int o = c ^ 1;
    U64 own = p.occ[c], ownp = p.bb[mk(c, PAWN)], back = BACKRANK[c];
    double v = 0;
    U64 minors = p.bb[mk(c, KNIGHT)] | p.bb[mk(c, BISHOP)];
    int developed = popcnt(minors & ~back);
    v += 10 * developed;
    int mob = 0;
    U64 mm = minors;
    while (mm) {
        int s = poplsb(mm);
        int pt = ptOf(p.sq[s]);
        int m = popcnt(attacksFrom(pt, s, p.all, c) & ~own);
        mob += m;
        if (!(back & BIT(s))) {
            if ((pt == BISHOP && m <= 2) || (pt == KNIGHT && (fileOf(s) == 0 || fileOf(s) == 7))) v -= 10;
        }
    }
    v += 1.5 * min(mob, 24);
    U64 queens = p.bb[mk(c, QUEEN)];
    int home = c == WHITE ? SQ_D1 : SQ_D8;
    if (queens && !(queens & BIT(home)) && developed < 2) v -= 15;
    for (int f = 0; f < 8; f += 7) {
        U64 b = ownp & FILE_BB[f];
        while (b) {
            int s = poplsb(b);
            int rel = c == WHITE ? rankOf(s) : 7 - rankOf(s);
            if (rel >= 2) v -= (rel == 2) ? 6 : 10;
        }
    }
    int mk_ = kingSq(p, c);
    int mkf = fileOf(mk_), mkr = rankOf(mk_);
    bool myCastled = mkr == (c == WHITE ? 0 : 7) && (mkf == 1 || mkf == 2 || mkf == 6 || mkf == 7);
    int myRights = c == WHITE ? (p.castling & 3) : (p.castling & 12);
    if (myCastled) v += 20;
    else if (!myRights) v -= 12;
    if (!myCastled) {
        U64 rk = p.bb[mk(c, ROOK)] & back;
        while (rk) { int s = poplsb(rk); if (fileOf(s) != 0 && fileOf(s) != 7) v -= 6; }
    }
    // afovoany
    static const int CS[4] = {SQ_D4, SQ_E4, SQ_D5, SQ_E5};
    for (int i = 0; i < 4; i++) {
        if (ownp & BIT(CS[i])) v += 14;
        v += 4 * min(popcnt(attackersTo(p, CS[i], c)), 3);
    }
    U64 pairA = c == WHITE ? BIT(SQ_D4) : BIT(SQ_D5), pairB = c == WHITE ? BIT(SQ_E4) : BIT(SQ_E5);
    if ((ownp & pairA) && (ownp & pairB)) v += 10;
    // fanakanana ny castling an'ny mpanohitra (ghost)
    int ok = kingSq(p, o), kf = fileOf(ok), kr = rankOf(ok), brank = o == WHITE ? 0 : 7;
    bool castled = kr == brank && (kf == 1 || kf == 2 || kf == 6 || kf == 7);
    if (!castled) {
        int oRights = o == WHITE ? (p.castling & 3) : (p.castling & 12);
        if (!oRights) {
            v += 40;
            if (kf != 4 || kr != brank) v += 15;
        } else {
            U64 attG = 0;                                   // ghost: ny piece-ko ihany
            for (int pt = PAWN; pt <= KING; pt++) {
                U64 b = p.bb[mk(c, pt)];
                while (b) { int s = poplsb(b); attG |= attacksFrom(pt, s, own, c); }
            }
            bool kside = o == WHITE ? (p.castling & 1) : (p.castling & 4);
            bool qside = o == WHITE ? (p.castling & 2) : (p.castling & 8);
            for (int side = 0; side < 2; side++) {
                if (!(side == 0 ? kside : qside)) continue;
                static const int KF[3] = {4, 5, 6}, QF[3] = {4, 3, 2};
                int real = 0, gh = 0;
                for (int i = 0; i < 3; i++) {
                    int sq = brank * 8 + (side == 0 ? KF[i] : QF[i]);
                    if (isAttacked(p, sq, c)) real++;
                    if (attG & BIT(sq)) gh++;
                }
                v += 10 * real + 4 * max(gh - real, 0);
            }
            if (p.stm == o && inCheck(p)) v += 20;
        }
        for (int f = kf - 1; f <= kf + 1; f++) {
            if (f < 0 || f > 7) continue;
            U64 fm = FILE_BB[f];
            if ((fm & (p.bb[mk(c, ROOK)] | p.bb[mk(c, QUEEN)])) && !(fm & ownp)) v += 14;
        }
        int weak = o == BLACK ? SQ_F7 : SQ_F2;
        if (kf == 4 && (p.bb[mk(o, PAWN)] & BIT(weak))) v += 5 * min(popcnt(attackersTo(p, weak, c)), 3);
    }
    return v;
}
static int openingTerms(const Position &p) {
    int d = (int)(openingSide(p, WHITE) - openingSide(p, BLACK));
    return max(-OPEN_CAP, min(OPEN_CAP, d));
}

static int evaluate(const Position &p) {
    int npmW = npmOf(p, WHITE), npmB = npmOf(p, BLACK);
    bool eg = (npmW + npmB) <= 2800;
    int score = p.score;
    score += positionalOf(p, eg, WHITE) - positionalOf(p, eg, BLACK);
    score += kingSafety(p, WHITE) - kingSafety(p, BLACK);
    score += (attackPower(p, WHITE) - attackPower(p, BLACK)) / 2;
    if (eg) score += endgameTerms(p, npmW, npmB);
    if (OPENING_ON && isOpeningPos(p, npmW, npmB)) score += openingTerms(p);
    return p.stm == WHITE ? score : -score;
}

// =====================================================================================
// SEARCH
// =====================================================================================
static const int MATE = 100000, INF = 1000000, MAXPLY = 100;
struct TTEntry { U64 key; Move move; int32_t score; int8_t depth; uint8_t flag; };
static vector<TTEntry> TT;
static size_t TTMASK = 0;
static void ttResize(size_t mb) {
    size_t n = 1;
    while (n * sizeof(TTEntry) * 2 <= mb * 1024 * 1024) n *= 2;
    TT.assign(n, TTEntry{0, 0, 0, 0, 0});
    TTMASK = n - 1;
}
static long long nodes = 0, nodeLimit = 0;
static bool stopSearch = false;
static chrono::steady_clock::time_point tStart;
static double timeLimitMs = 0;
static Move killers[MAXPLY + 4][2];
static int historyT[2][64][64];
static Move rootBest = 0;

static double elapsedMs() { return chrono::duration<double, milli>(chrono::steady_clock::now() - tStart).count(); }
static inline void checkTime() {
    if ((timeLimitMs > 0 && elapsedMs() >= timeLimitMs) || (nodeLimit > 0 && nodes >= nodeLimit)) stopSearch = true;
}
static inline int scoreToTT(int s, int ply) { return s > MATE - 1000 ? s + ply : (s < -MATE + 1000 ? s - ply : s); }
static inline int scoreFromTT(int s, int ply) { return s > MATE - 1000 ? s - ply : (s < -MATE + 1000 ? s + ply : s); }

static bool insufficientMaterial(const Position &p) {
    if (p.bb[0] | p.bb[6] | p.bb[3] | p.bb[9] | p.bb[4] | p.bb[10]) return false;
    return popcnt(p.bb[1] | p.bb[2] | p.bb[7] | p.bb[8]) <= 1;
}
static bool isDrawPos(const Position &p) {
    if (p.halfmove >= 100) return true;
    int n = min(p.halfmove, p.hply);
    for (int i = 2; i <= n; i += 2)
        if (p.hist[p.hply - i].key == p.key) return true;
    return insufficientMaterial(p);
}

static void scoreMoves(const Position &p, MoveList &l, Move ttMove, int ply) {
    for (int i = 0; i < l.n; i++) {
        Move m = l.m[i];
        int sc;
        if (m == ttMove) sc = 1000000;
        else if (m & (F_CAP | F_EP)) {
            int victim = (m & F_EP) ? PAWN : ptOf(p.sq[mTo(m)]);
            int attacker = ptOf(p.sq[mFrom(m)]);
            sc = 100000 + 10 * VALS[victim] - VALS[attacker] + (mPromo(m) ? VALS[mPromo(m)] : 0);
        } else if (mPromo(m)) sc = 90000 + VALS[mPromo(m)];
        else if (m == killers[ply][0]) sc = 80000;
        else if (m == killers[ply][1]) sc = 79000;
        else sc = historyT[p.stm][mFrom(m)][mTo(m)];
        l.sc[i] = sc;
    }
}
static inline void pickMove(MoveList &l, int i) {
    int best = i;
    for (int j = i + 1; j < l.n; j++) if (l.sc[j] > l.sc[best]) best = j;
    if (best != i) { swap(l.m[i], l.m[best]); swap(l.sc[i], l.sc[best]); }
}

static int quiesce(Position &p, int alpha, int beta, int ply) {
    if ((++nodes & 2047) == 0) checkTime();
    if (stopSearch) return 0;
    if (ply >= MAXPLY) return evaluate(p);
    int stand = evaluate(p);
    if (stand >= beta) return stand;
    if (stand > alpha) alpha = stand;
    MoveList l; genMoves(p, l, true);
    scoreMoves(p, l, 0, ply);
    int best = stand;
    for (int i = 0; i < l.n; i++) {
        pickMove(l, i);
        Move m = l.m[i];
        if (!mPromo(m)) {
            int victim = (m & F_EP) ? PAWN : ptOf(p.sq[mTo(m)]);
            if (stand + VALS[victim] + 200 < alpha) continue;
        }
        if (!makeMove(p, m)) continue;
        int v = -quiesce(p, -beta, -alpha, ply + 1);
        unmakeMove(p);
        if (stopSearch) return 0;
        if (v > best) { best = v; if (v > alpha) { alpha = v; if (v >= beta) return v; } }
    }
    return best;
}

static int search(Position &p, int depth, int alpha, int beta, int ply, bool allowNull) {
    if ((++nodes & 2047) == 0) checkTime();
    if (stopSearch) return 0;
    bool root = ply == 0, pvNode = (beta - alpha) > 1;
    if (!root) {
        if (isDrawPos(p)) return 0;
        alpha = max(alpha, -MATE + ply);
        beta = min(beta, MATE - ply - 1);
        if (alpha >= beta) return alpha;
        if (ply >= MAXPLY) return evaluate(p);
    }
    bool chk = inCheck(p);
    if (chk) depth++;
    if (depth <= 0) return quiesce(p, alpha, beta, ply);

    TTEntry &e = TT[p.key & TTMASK];
    Move ttMove = 0;
    if (e.key == p.key) {
        ttMove = e.move;
        if (!pvNode && !root && e.depth >= depth) {
            int s = scoreFromTT(e.score, ply);
            if (e.flag == 0) return s;
            if (e.flag == 1 && s >= beta) return s;
            if (e.flag == 2 && s <= alpha) return s;
        }
    }
    int staticEval = 0;
    bool haveEval = false;
    if (!pvNode && !chk) { staticEval = evaluate(p); haveEval = true; }
    // reverse futility
    if (haveEval && depth <= 3 && staticEval - 120 * depth >= beta && beta < MATE - 1000) return staticEval;
    // null move
    if (allowNull && haveEval && depth >= 3 && staticEval >= beta && beta < MATE - 1000 &&
        (p.occ[p.stm] & ~(p.bb[mk(p.stm, PAWN)] | p.bb[mk(p.stm, KING)]))) {
        int R = 2 + depth / 6;
        makeNull(p);
        int v = -search(p, depth - 1 - R, -beta, -beta + 1, ply + 1, false);
        unmakeNull(p);
        if (stopSearch) return 0;
        if (v >= beta) return v >= MATE - 1000 ? beta : v;
    }
    MoveList l; genMoves(p, l, false);
    scoreMoves(p, l, ttMove, ply);
    int best = -INF, origAlpha = alpha, legal = 0;
    Move bestMove = 0;
    Move quietsTried[64]; int nq = 0;
    for (int i = 0; i < l.n; i++) {
        pickMove(l, i);
        Move m = l.m[i];
        bool quiet = !(m & (F_CAP | F_EP)) && !mPromo(m);
        if (!makeMove(p, m)) continue;
        legal++;
        bool givesCheck = inCheck(p);
        int newDepth = depth - 1, v;
        if (legal == 1) v = -search(p, newDepth, -beta, -alpha, ply + 1, true);
        else {
            int red = 0;
            if (depth >= 3 && legal >= 4 && quiet && !chk && !givesCheck) {
                red = 1 + (legal >= 8 && depth >= 5) + (legal >= 14 && depth >= 7);
                if (red > newDepth - 1) red = max(0, newDepth - 1);
            }
            v = -search(p, newDepth - red, -alpha - 1, -alpha, ply + 1, true);
            if (v > alpha && red) v = -search(p, newDepth, -alpha - 1, -alpha, ply + 1, true);
            if (v > alpha && v < beta) v = -search(p, newDepth, -beta, -alpha, ply + 1, true);
        }
        unmakeMove(p);
        if (stopSearch) return 0;
        if (quiet && nq < 64) quietsTried[nq++] = m;
        if (v > best) {
            best = v; bestMove = m;
            if (v > alpha) {
                alpha = v;
                if (root) rootBest = m;
                if (v >= beta) {
                    if (quiet) {
                        if (killers[ply][0] != m) { killers[ply][1] = killers[ply][0]; killers[ply][0] = m; }
                        int bonus = depth * depth;
                        historyT[p.stm][mFrom(m)][mTo(m)] += bonus;
                        for (int k = 0; k < nq - 1; k++) {
                            int &h = historyT[p.stm][mFrom(quietsTried[k])][mTo(quietsTried[k])];
                            h -= bonus / 2; if (h < -50000) h = -50000;
                        }
                        if (historyT[p.stm][mFrom(m)][mTo(m)] > 70000) {
                            for (int a = 0; a < 2; a++) for (int b = 0; b < 64; b++) for (int c = 0; c < 64; c++) historyT[a][b][c] /= 2;
                        }
                    }
                    break;
                }
            }
        }
    }
    if (legal == 0) return chk ? -MATE + ply : 0;
    e.key = p.key; e.move = bestMove; e.depth = (int8_t)depth;
    e.flag = best >= beta ? 1 : (alpha > origAlpha ? 0 : 2);
    e.score = scoreToTT(best, ply);
    return best;
}

static string pvString(Position &p, int maxLen) {
    string s;
    int made = 0;
    for (int i = 0; i < maxLen; i++) {
        TTEntry &e = TT[p.key & TTMASK];
        if (e.key != p.key || !e.move) break;
        Move m = e.move;
        MoveList l; genMoves(p, l, false);
        bool found = false;
        for (int k = 0; k < l.n; k++) if (l.m[k] == m) { found = true; break; }
        if (!found) break;
        string ms = moveStr(m);
        if (!makeMove(p, m)) break;
        made++;
        s += ms + " ";
    }
    while (made--) unmakeMove(p);
    return s;
}

// ---- iterative deepening
struct Limits { int depth = 64; double movetime = 0; long long nodes = 0; bool infinite = false; };
static bool quietOut = false;

static Move findMoveStr(Position &p, const string &s) {
    MoveList l; genMoves(p, l, false);
    for (int i = 0; i < l.n; i++) {
        if (moveStr(l.m[i]) == s) {
            if (makeMove(p, l.m[i])) { unmakeMove(p); return l.m[i]; }
        }
    }
    return 0;
}

static Move think(Position &p, const Limits &lim, int *outScore = nullptr, int *outDepth = nullptr) {
    nodes = 0; nodeLimit = lim.nodes; stopSearch = false; timeLimitMs = lim.movetime;
    tStart = chrono::steady_clock::now();
    memset(killers, 0, sizeof killers);
    memset(historyT, 0, sizeof historyT);
    MoveList root; genMoves(p, root, false);
    // misy move ara-dalàna ve?
    Move firstLegal = 0; int nLegal = 0;
    for (int i = 0; i < root.n; i++) if (makeMove(p, root.m[i])) { unmakeMove(p); if (!firstLegal) firstLegal = root.m[i]; nLegal++; }
    if (!nLegal) return 0;
    Move best = firstLegal; int bestScore = 0, doneDepth = 0;
    if (nLegal == 1 && lim.movetime > 0) { if (outScore) *outScore = 0; if (outDepth) *outDepth = 0; return best; }
    for (int depth = 1; depth <= lim.depth; depth++) {
        rootBest = best;
        int score = search(p, depth, -INF, INF, 0, false);
        if (stopSearch) { if (rootBest) best = rootBest; break; }
        best = rootBest ? rootBest : best;
        bestScore = score; doneDepth = depth;
        if (!quietOut) {
            double t = elapsedMs();
            string pv = pvString(p, depth + 4);
            printf("info depth %d score cp %d nodes %lld nps %lld time %d pv %s\n", depth, score, nodes,
                   (long long)(nodes / max(t, 1.0) * 1000), (int)t, pv.c_str());
            fflush(stdout);
        }
        if (abs(score) > MATE - 200) break;
        if (lim.movetime > 0 && elapsedMs() > lim.movetime * 0.6) break;   // tsy ho ampy ho an'ny depth manaraka
    }
    if (outScore) *outScore = bestScore;
    if (outDepth) *outDepth = doneDepth;
    return best;
}

// ---- UCI
static Position gpos;
static void setPosition(const string &line) {
    istringstream is(line);
    string tok;
    is >> tok;                       // position
    is >> tok;
    if (tok == "startpos") setFen(gpos, STARTFEN);
    else if (tok == "fen") {
        string fen, t;
        while (is >> t && t != "moves") fen += t + " ";
        setFen(gpos, fen);
        if (t == "moves") {
            while (is >> t) { Move m = findMoveStr(gpos, t); if (m && makeMove(gpos, m)) {} }
        }
        return;
    }
    if (is >> tok && tok == "moves") {
        while (is >> tok) { Move m = findMoveStr(gpos, tok); if (m) makeMove(gpos, m); }
    }
}

static void goCommand(const string &line) {
    istringstream is(line);
    string tok;
    is >> tok;
    Limits lim;
    long long wtime = -1, btime = -1, winc = 0, binc = 0, movestogo = 0;
    while (is >> tok) {
        if (tok == "depth") is >> lim.depth;
        else if (tok == "movetime") { long long t; is >> t; lim.movetime = (double)t; }
        else if (tok == "nodes") is >> lim.nodes;
        else if (tok == "wtime") is >> wtime;
        else if (tok == "btime") is >> btime;
        else if (tok == "winc") is >> winc;
        else if (tok == "binc") is >> binc;
        else if (tok == "movestogo") is >> movestogo;
        else if (tok == "infinite") lim.infinite = true;
    }
    if (lim.movetime == 0 && !lim.nodes && lim.depth == 64) {
        long long myt = gpos.stm == WHITE ? wtime : btime, inc = gpos.stm == WHITE ? winc : binc;
        if (myt >= 0) {
            long long mtg = movestogo > 0 ? movestogo : 30;
            lim.movetime = max(20.0, min((double)myt / mtg + inc * 0.8, myt * 0.5));
        } else if (!lim.infinite) lim.movetime = 1000;
    }
    int sc, dp;
    Move m = think(gpos, lim, &sc, &dp);
    printf("bestmove %s\n", m ? moveStr(m).c_str() : "0000");
    fflush(stdout);
}

static const char *BENCH_FENS[] = {
    STARTFEN,
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
    "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
    "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
    "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
    "6k1/5ppp/8/8/8/8/5PPP/3R2K1 w - - 0 1",
};

int main(int argc, char **argv) {
    initTables();
    BACKRANK[WHITE] = RANK_BB[0]; BACKRANK[BLACK] = RANK_BB[7];
    ttResize(64);
    setFen(gpos, STARTFEN);
    string line;
    while (getline(cin, line)) {
        if (line.empty()) continue;
        istringstream is(line);
        string cmd;
        is >> cmd;
        if (cmd == "uci") {
            printf("id name JORO-cpp 0.2\nid author JORO\noption name Hash type spin default 64 min 1 max 4096\noption name Opening type check default false\noption name OpenCap type spin default 100 min 0 max 400\nuciok\n");
            fflush(stdout);
        } else if (cmd == "isready") { printf("readyok\n"); fflush(stdout); }
        else if (cmd == "ucinewgame") { setFen(gpos, STARTFEN); fill(TT.begin(), TT.end(), TTEntry{0, 0, 0, 0, 0}); }
        else if (cmd == "setoption") {
            string t, name, val;
            is >> t;                                          // "name"
            while (is >> t && t != "value") name += (name.empty() ? "" : " ") + t;
            while (is >> t) val += (val.empty() ? "" : " ") + t;
            if (name == "Hash") ttResize(max(1, atoi(val.c_str())));
            else if (name == "Opening") OPENING_ON = (val == "true" || val == "1");
            else if (name == "OpenCap") OPEN_CAP = max(0, atoi(val.c_str()));
        } else if (cmd == "position") setPosition(line);
        else if (cmd == "go") goCommand(line);
        else if (cmd == "perft") {
            int d = 5; is >> d;
            auto t0 = chrono::steady_clock::now();
            U64 n = perft(gpos, d);
            double s = chrono::duration<double>(chrono::steady_clock::now() - t0).count();
            printf("perft %d = %llu (%.2fs)\n", d, (unsigned long long)n, s); fflush(stdout);
        } else if (cmd == "eval") { printf("eval (stm) = %d\n", evaluate(gpos)); fflush(stdout); }
        else if (cmd == "bench") {
            int d = 9; is >> d;
            long long total = 0; double tt = 0;
            quietOut = true;
            for (const char *f : BENCH_FENS) {
                setFen(gpos, f);
                fill(TT.begin(), TT.end(), TTEntry{0, 0, 0, 0, 0});
                Limits lim; lim.depth = d;
                int sc, dp;
                think(gpos, lim, &sc, &dp);
                total += nodes; tt += elapsedMs();
            }
            quietOut = false;
            printf("bench depth %d: nodes %lld time %.0f ms nps %.0f\n", d, total, tt, total / max(tt, 1.0) * 1000);
            fflush(stdout);
        } else if (cmd == "quit") break;
    }
    return 0;
}
