// JORO C++ core v0.1  -  bitboard engine (UCI) : move generator, search, evaluation lite.
// v0.7: GHOST afindra tanteraka ao amin'ny SEARCH ; REPERTOIRE ihany (KG/Italian/Scotch - Sicilian/KID/French) ; OPENING SCHOOL (5 zavatra).
// v0.6: ATTACK RACE (ETA), CLEAN MOVE (search), OPENING 2 (castle prediction, Right/Wrong castle, passive dev).
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
static string fenStr(const Position &p) {
    string o;
    static const char *PCH = "PNBRQKpnbrqk";
    for (int r = 7; r >= 0; r--) {
        int e = 0;
        for (int f = 0; f < 8; f++) {
            int pc = p.sq[r * 8 + f];
            if (pc == NOPIECE) { e++; continue; }
            if (e) { o += char('0' + e); e = 0; }
            o += PCH[pc];
        }
        if (e) o += char('0' + e);
        if (r) o += '/';
    }
    o += p.stm == WHITE ? " w " : " b ";
    string c;
    if (p.castling & 1) c += 'K';
    if (p.castling & 2) c += 'Q';
    if (p.castling & 4) c += 'k';
    if (p.castling & 8) c += 'q';
    o += c.empty() ? "-" : c;
    o += " ";
    if (p.ep >= 0) { o += char('a' + fileOf(p.ep)); o += char('1' + rankOf(p.ep)); } else o += "-";
    o += " " + to_string(p.halfmove) + " " + to_string(p.fullmove);
    return o;
}
static int skippedMoves = 0;      // move tsy ara-dalàna tao amin'ny "position ... moves" (fanamarinana FEN)
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
// castling ho an'ny king-ko, ary FANAKANANA ny castling an'ny mpanohitra (real attack ihany ao amin'ny eval ;
// ny ghost dia ao amin'ny search : castleDenialGain).
// =====================================================================================
static bool OPENING_ON = true;
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
// ---- OPENING 2 : anticipation castling, Right/Wrong Castle, Passive Development
static bool OPEN2_ON = true;
static int OPEN2_CAP = 60;
static const int OPEN2_NPM = 3600;                 // manana ihany hatramin'ny fiafaran'ny opening/middlegame aloha
enum { CP_NONE = 0, CP_SHORT, CP_LONG, CP_CENTER };
static inline int relSq(int c, int f, int rr) { return (c == WHITE ? rr : 7 - rr) * 8 + f; }
static inline bool isCastled(const Position &p, int c) {
    int k = kingSq(p, c), kf = fileOf(k);
    return rankOf(k) == (c == WHITE ? 0 : 7) && (kf == 1 || kf == 2 || kf == 6 || kf == 7);
}
// kalitaon'ny Pawn Shield (side 0 = short f,g,h ; 1 = long a,b,c). Avo = Top Defense, ambany = Wrong Castle.
static int shieldQuality(const Position &p, int c, int side) {
    int o = c ^ 1, f0 = side == 0 ? 5 : 0, q = 0;
    U64 ownp = p.bb[mk(c, PAWN)], oppp = p.bb[mk(o, PAWN)];
    for (int f = f0; f < f0 + 3; f++) {
        int rel = -1;
        U64 b = ownp & FILE_BB[f];
        while (b) { int s = poplsb(b); int r = c == WHITE ? rankOf(s) : 7 - rankOf(s); if (rel < 0 || r < rel) rel = r; }
        if (rel < 0) q -= 3;                 // colonne misokatra
        else if (rel == 1) q += 2;           // pawn eo an-trano
        else if (rel == 2) q += 1;           // nandroso indray mandeha (sembana kely)
        else q -= 1;                         // nandroso be (lavaka)
        U64 e = oppp & FILE_BB[f];
        while (e) { int s = poplsb(e); int r = c == WHITE ? rankOf(s) : 7 - rankOf(s); if (r <= 2) q -= 2; else if (r <= 3) q -= 1; }   // Pawn Storm
    }
    if (side == 0) {
        if ((p.bb[mk(c, BISHOP)] & BIT(relSq(c, 6, 1))) && (ownp & BIT(relSq(c, 6, 2)))) q += 1;   // fianchetto
        if (p.bb[mk(c, KNIGHT)] & BIT(relSq(c, 5, 2))) q += 1;                                      // Nf3/Nf6
    } else {
        if ((p.bb[mk(c, BISHOP)] & BIT(relSq(c, 1, 1))) && (ownp & BIT(relSq(c, 1, 2)))) q += 1;
        if (p.bb[mk(c, KNIGHT)] & BIT(relSq(c, 2, 2))) q += 1;
    }
    return max(-9, min(7, q));
}
static int castleReadiness(const Position &p, int c, int side) {      // 0..100 : pièces efa niala teo anelanelan'ny king sy rook
    static const int SQS[2][3] = {{5, 6, -1}, {1, 2, 3}};
    int n = side == 0 ? 2 : 3, e = 0;
    for (int i = 0; i < n; i++) if (!(p.all & BIT(relSq(c, SQS[side][i], 0)))) e++;
    return 100 * e / n;
}
static int predictCastle(const Position &p, int c) {
    int br = c == WHITE ? 0 : 7, k = kingSq(p, c), kf = fileOf(k), kr = rankOf(k);
    if (isCastled(p, c)) return kf >= 5 ? CP_SHORT : CP_LONG;
    int rights = c == WHITE ? (p.castling & 3) : (p.castling & 12);
    if (!rights) return (kr == br && (kf == 3 || kf == 4)) ? CP_CENTER : CP_NONE;
    bool ks = c == WHITE ? (p.castling & 1) : (p.castling & 4), qs = c == WHITE ? (p.castling & 2) : (p.castling & 8);
    int rs = ks ? castleReadiness(p, c, 0) : -1, rl = qs ? castleReadiness(p, c, 1) : -1;
    return rl > rs ? CP_LONG : CP_SHORT;
}
// Passive Development : bishop/knight mijanona eo anoloan'ny pawn afovoany tsy mbola nihetsika, sns.
static int passiveDev(const Position &p, int c) {
    U64 ownp = p.bb[mk(c, PAWN)];
    int pen = 0;
    for (int f = 3; f <= 4; f++) {
        if (ownp & BIT(relSq(c, f, 1))) {
            int pc = p.sq[relSq(c, f, 2)];
            if (pc != NOPIECE && colOf(pc) == c) {
                if (ptOf(pc) == BISHOP) pen += 14; else if (ptOf(pc) == KNIGHT) pen += 8;
            }
        }
    }
    bool centreOpen = (ownp & (BIT(relSq(c, 3, 3)) | BIT(relSq(c, 4, 3)))) != 0;     // d4/e4 (d5/e5)
    if (!centreOpen)
        for (int f = 3; f <= 4; f++) if (p.sq[relSq(c, f, 2)] == mk(c, BISHOP)) pen += 8;   // Bd3/Be3 (Bd6/Be6) alohan'ny fanokafana
    for (int f = 2; f <= 5; f += 3) {                                                   // bishop c1/f1 voasakana ny knight-ny
        if (!(p.bb[mk(c, BISHOP)] & BIT(relSq(c, f, 0)))) continue;
        for (int df = -1; df <= 1; df += 2) if (p.sq[relSq(c, f + df, 1)] == mk(c, KNIGHT)) pen += 5;
    }
    return min(pen, 40);
}
static int castleStructSide(const Position &p, int c) {
    int pc = predictCastle(p, c);
    if (pc != CP_SHORT && pc != CP_LONG) return 0;
    bool castled = isCastled(p, c);
    int q = shieldQuality(p, c, pc == CP_LONG ? 1 : 0);
    int v = (q - 4) * (castled ? 5 : 3);
    if (q >= 6) v += castled ? 10 : 5;       // Right Castle
    if (q <= 1) v -= castled ? 15 : 8;       // Wrong Castle
    return v;
}
static int castleStructTerms(const Position &p) {
    int d = castleStructSide(p, WHITE) - castleStructSide(p, BLACK);
    return max(-OPEN2_CAP, min(OPEN2_CAP, d));
}

// ---- OPPOSITE-SIDE CASTLING + PAWN STORM
// castleCommit : -1 = tsy mbola voafidy ; 0 = short ; 1 = long (efa nanao castle, na efa vonona ho amin'ny elatra iray)
static bool STORM_ON = true;
static int STORM_CAP = 80;
static int SAMESIDE_PEN = 35;
static int ROOT_US = -1;                 // ny lafy mijery amin'ny root (JORO) : ny safidy castling dia ho an'i JORO ihany
static int castleCommit(const Position &p, int c) {
    if (isCastled(p, c)) return fileOf(kingSq(p, c)) >= 5 ? 0 : 1;
    int rights = c == WHITE ? (p.castling & 3) : (p.castling & 12);
    if (!rights) return -1;
    bool ks = c == WHITE ? (p.castling & 1) : (p.castling & 4), qs = c == WHITE ? (p.castling & 2) : (p.castling & 8);
    int rs = ks ? castleReadiness(p, c, 0) : -1, rl = qs ? castleReadiness(p, c, 1) : -1;
    if (max(rs, rl) < 50 || rs == rl) return -1;
    return rl > rs ? 1 : 0;
}
// Raha short castle ny mpanohitra -> tsy short ihany koa i JORO (long / center) ; raha long -> tsy long (safe, short).
// Ho an'i c fotsiny (c = JORO) : ny sazy/bonus dia mitombo rehefa efa tena nanao castle ny roa tonta.
static int castleVsOpp(const Position &p, int c) {
    int mine = castleCommit(p, c), th = castleCommit(p, c ^ 1);
    if (mine < 0 || th < 0) return 0;
    double w = 0.5 + 0.25 * isCastled(p, c) + 0.25 * isCastled(p, c ^ 1);
    return mine == th ? -(int)(SAMESIDE_PEN * w) : (int)(10 * w);
}
// Pawn Storm / fanamelohana ny shield aorian'ny castle : pawn manafika ny elatra, lever, colonne misokatra, piece rehetra manindry.
static int stormSide(const Position &p, int c) {
    int o = c ^ 1, so = castleCommit(p, o);
    if (so < 0) return 0;
    double factor = castleCommit(p, c) == so ? 0.4 : 1.0;          // castle mitovy elatra -> storm malemy kokoa (loza ho an'ny king-ko)
    int f0 = so == 0 ? 5 : 0;
    U64 wingFiles = FILE_BB[f0] | FILE_BB[f0 + 1] | FILE_BB[f0 + 2];
    U64 ownp = p.bb[mk(c, PAWN)], oppp = p.bb[mk(o, PAWN)], shield = oppp & wingFiles;
    int v = 0;
    U64 b = ownp & wingFiles;
    while (b) {
        int s = poplsb(b), r = c == WHITE ? rankOf(s) : 7 - rankOf(s);
        if (r >= 3) v += min(15, 5 * (r - 2));                        // pawn nandroso
        if (PAWN_ATT[c][s] & shield) v += 6;                          // lever manohitra ny shield
    }
    int eks = kingSq(p, o), kf = fileOf(eks);
    for (int f = max(0, kf - 1); f <= min(7, kf + 1); f++) {          // colonne mankany amin'ny king
        U64 fm = FILE_BB[f];
        if (fm & ownp) continue;
        if (fm & (p.bb[mk(c, ROOK)] | p.bb[mk(c, QUEEN)])) v += (fm & oppp) ? 8 : 14;
        else if (!(fm & oppp)) v += 4;
    }
    U64 zone = KING_ATT[eks] | BIT(eks);
    for (int ff = f0; ff < f0 + 3; ff++) for (int rr = 0; rr < 3; rr++) zone |= BIT(relSq(o, ff, rr));
    int press = 0;
    for (int pt = KNIGHT; pt <= QUEEN; pt++) {
        U64 pcs = p.bb[mk(c, pt)];
        while (pcs) { int s = poplsb(pcs); press += ATTW[pt] * popcnt(attacksFrom(pt, s, p.all, c) & zone); }
    }
    v += min(24, 2 * press);                                          // piece rehetra manindry
    return (int)(min(v, 70) * factor);
}
static int stormTerms(const Position &p) {
    int d = stormSide(p, WHITE) - stormSide(p, BLACK);
    return max(-STORM_CAP, min(STORM_CAP, d));
}

static double openingSide(const Position &p, int c) {
    int o = c ^ 1;
    U64 own = p.occ[c], ownp = p.bb[mk(c, PAWN)], back = BACKRANK[c];
    double v = 0;
    U64 minors = p.bb[mk(c, KNIGHT)] | p.bb[mk(c, BISHOP)];
    int developed = popcnt(minors & ~back);
    v += 10 * developed;
    if (OPEN2_ON) v -= passiveDev(p, c);
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
    // fanakanana ny castling an'ny mpanohitra (real attack ; ny ghost dia ao amin'ny search)
    int ok = kingSq(p, o), kf = fileOf(ok), kr = rankOf(ok), brank = o == WHITE ? 0 : 7;
    bool castled = kr == brank && (kf == 1 || kf == 2 || kf == 6 || kf == 7);
    if (!castled) {
        int oRights = o == WHITE ? (p.castling & 3) : (p.castling & 12);
        if (!oRights) {
            v += 40;
            if (kf != 4 || kr != brank) v += 15;
        } else {
            bool kside = o == WHITE ? (p.castling & 1) : (p.castling & 4);
            bool qside = o == WHITE ? (p.castling & 2) : (p.castling & 8);
            for (int side = 0; side < 2; side++) {
                if (!(side == 0 ? kside : qside)) continue;
                static const int KF[3] = {4, 5, 6}, QF[3] = {4, 3, 2};
                int real = 0;
                for (int i = 0; i < 3; i++) {
                    int sq = brank * 8 + (side == 0 ? KF[i] : QF[i]);
                    if (isAttacked(p, sq, c)) real++;
                }
                v += 10 * real;
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


// =====================================================================================
// OPENING SCHOOL - ny filozofian'i JORO :
//   « Ny opening dia tsy natao hahazoana toerana tsara fotsiny, fa natao hanomanana ny fandravana
//     ny fiarovan'ny King adversaire. »
// Tsy misy opening mandresy ho azy : ny repertoire dia mahomby rehefa fantatra ny plans sy ny tactical patterns ao anatiny.
// Zavatra 5 tokony hampianarina mandritra ny Opening :
//   1. King Weakness Detection        -> kingWeakness()  (eval)
//   2. Attack Preparation             -> attackPrep()    (eval)
//   3. Pawn Structure Intelligence    -> pawnIntel()     (eval)
//   4. Sacrifice Evaluation           -> sacComp() (eval : compensation) + sacrificeVerdict() (search : mendrika / tsy mendrika)
//   5. Opening-to-Middlegame Transition -> transition()  (eval)
// =====================================================================================
static bool SCHOOL_ON = true;
static int SCHOOL_CAP = 90;
static const int SCHOOL_NPM = 4200;            // manomboka amin'ny opening ka hatramin'ny middlegame aloha

static U64 attackedBy(const Position &p, int c) {
    U64 a = 0;
    for (int pt = PAWN; pt <= QUEEN; pt++) {
        U64 b = p.bb[mk(c, pt)];
        while (b) a |= attacksFrom(pt, poplsb(b), p.all, c);
    }
    return a;
}
static inline int matOf(const Position &p, int c) {
    int m = 0;
    for (int pt = PAWN; pt <= QUEEN; pt++) m += VALS[pt] * popcnt(p.bb[mk(c, pt)]);
    return m;
}
// ny faritra manodidina ny king-n'i o (3 colonne x 3 rang) + ny king zone : ny toerana hanatanterahana ny attack
static U64 wingZone(const Position &p, int o) {
    int ek = kingSq(p, o), kf = fileOf(ek);
    U64 w = KING_ATT[ek] | BIT(ek);
    for (int f = max(0, kf - 1); f <= min(7, kf + 1); f++)
        for (int rr = 0; rr < 3; rr++) w |= BIT(relSq(o, f, rr));
    return w;
}

// 1. KING WEAKNESS DETECTION : fahalemena manodidina ny king-n'i (c^1), ho an'ny mpanafika c
static int kingWeakness(const Position &p, int c) {
    int o = c ^ 1, ek = kingSq(p, o), kf = fileOf(ek), kr = rankOf(ek), br = o == WHITE ? 0 : 7;
    U64 ownp = p.bb[mk(o, PAWN)], allp = p.bb[mk(WHITE, PAWN)] | p.bb[mk(BLACK, PAWN)];
    U64 zone = KING_ATT[ek] | BIT(ek), atk = attackedBy(p, c);
    int w = 0;
    bool castled = isCastled(p, o);
    if (!castled) {
        int rights = o == WHITE ? (p.castling & 3) : (p.castling & 12);
        if (kr != br) w += 25;                                   // king nandeha anaty afovoany
        else if (kf == 3 || kf == 4) w += rights ? 8 : 20;       // mbola any afovoany (tsy afaka mandray castle = ratsy kokoa)
    }
    for (int f = max(0, kf - 1); f <= min(7, kf + 1); f++) {     // Pawn Shield : tsy ampy / nandroso loatra
        int rel = -1;
        U64 b = ownp & FILE_BB[f];
        while (b) { int s = poplsb(b); int r = o == WHITE ? rankOf(s) : 7 - rankOf(s); if (rel < 0 || r < rel) rel = r; }
        if (rel < 0) w += (allp & FILE_BB[f]) ? 8 : 12;          // colonne misokatra (12) / semi-open (8)
        else if (rel == 2) w += 3;
        else if (rel >= 3) w += 6;
    }
    U64 pa = 0, b2 = ownp;
    while (b2) pa |= PAWN_ATT[o][poplsb(b2)];
    w += 2 * min(popcnt(zone & ~pa & ~p.occ[o] & atk), 5);       // lavaka (toerana tsy arovan'ny pawn) izay tratry ny piece-ko
    if (!castled && kf == 4) {                                   // f2/f7 : arovan'ny king ihany
        int sq = relSq(o, 5, 1);
        if (ownp & BIT(sq)) w += 4 * min(popcnt(attackersTo(p, sq, c)), 3);
    }
    return min(w, 60);
}

// 2. ATTACK PREPARATION : manangona pièces maromaro alohan'ny hanaovana attaque
static int attackPrep(const Position &p, int c) {
    int o = c ^ 1, n = 0, dev = undeveloped(p, c);
    U64 wing = wingZone(p, o);
    bool queenIn = false;
    for (int pt = KNIGHT; pt <= QUEEN; pt++) {
        U64 b = p.bb[mk(c, pt)];
        while (b) {
            int s = poplsb(b);
            bool home = rankOf(s) == (c == WHITE ? 0 : 7);
            if (home && pt != ROOK && pt != QUEEN) continue;     // tsy mbola nivoaka
            if (attacksFrom(pt, s, p.all, c) & wing) { n++; if (pt == QUEEN) queenIn = true; }
        }
    }
    int v = 0;
    if (n >= 3) v = 18 + 8 * min(n - 3, 3);
    else if (n == 2) v = 8;
    int pawnsAdv = 0;
    U64 b = p.bb[mk(c, PAWN)];
    while (b) { int s = poplsb(b); int r = c == WHITE ? rankOf(s) : 7 - rankOf(s); if (r >= 3 && (wing & FILE_BB[fileOf(s)])) pawnsAdv++; }
    U64 q = p.bb[mk(c, QUEEN)];
    bool qOut = q && !(q & BIT(c == WHITE ? 3 : 59));
    if (queenIn && qOut && (n - 1) < 2 && dev >= 2) v -= 15;     // dame mitokana : attaque tsy vita
    if (pawnsAdv >= 2 && n < 2) v -= 8;                          // pawn storm tsy misy piece
    if (dev == 0 && n >= 2) v += 6;
    return max(-25, min(40, v));
}

// 3. PAWN STRUCTURE INTELLIGENCE : mamantatra ny pawn tokony hotafihina (isolated, doubled, backward, f2/f7)
static int pawnIntel(const Position &p, int c) {
    int o = c ^ 1, total = 0;
    U64 op = p.bb[mk(o, PAWN)], b = op;
    bool castled = isCastled(p, o);
    while (b) {
        int s = poplsb(b), f = fileOf(s), r = o == WHITE ? rankOf(s) : 7 - rankOf(s);
        U64 adj = ((f > 0 ? FILE_BB[f - 1] : 0) | (f < 7 ? FILE_BB[f + 1] : 0)) & op;
        int wgt = 0;
        if (!adj) wgt += 6;                                       // isolated
        else {
            int minRel = 8;
            U64 a = adj;
            while (a) { int s2 = poplsb(a); int r2 = o == WHITE ? rankOf(s2) : 7 - rankOf(s2); minRel = min(minRel, r2); }
            if (minRel > r && r >= 1) wgt += 4;                   // backward
        }
        if (popcnt(op & FILE_BB[f]) > 1) wgt += 3;                // doubled
        if (!castled && f == 5 && r == 1) wgt += 5;               // f2/f7 alohan'ny castle
        if (!wgt) continue;
        int att = popcnt(attackersTo(p, s, c)), def = popcnt(attackersTo(p, s, o));
        if (att) total += wgt * (1 + min(att, 3)) / 2 + (att > def ? wgt : 0);
    }
    return min(total, 40);
}

// 4. SACRIFICE EVALUATION (eval) : raha efa very pawn/pièce, ny vola mendrika raha lehibe ny fahalemena + ny attack vonona
static int sacComp(const Position &p, int c, int weak, int prep) {
    int diff = matOf(p, c) - matOf(p, c ^ 1);
    if (diff > -60 || diff < -320) return 0;
    int need = diff < -150 ? 55 : 35, tot = weak + prep;
    if (tot < need) return 0;
    return min(45, tot * 2 / 3);
}

// 5. OPENING-TO-MIDDLEGAME TRANSITION : tsy mijanona amin'ny avantage ; rehefa vita ny fivoarana dia attaque tena izy
static int transition(const Position &p, int c) {
    int o = c ^ 1, dev = undeveloped(p, c), devO = undeveloped(p, o);
    if (dev > 1) return 0;                                       // mbola opening
    U64 wing = wingZone(p, o);
    int aim = 0;
    for (int pt = KNIGHT; pt <= QUEEN; pt++) {
        U64 b = p.bb[mk(c, pt)];
        while (b) if (attacksFrom(pt, poplsb(b), p.all, c) & wing) aim++;
    }
    int v = aim >= 2 ? min(30, 8 * (aim - 1)) : -10;             // -10 = avantage tsy novaina ho attaque
    if (devO > dev) v += 4 * (devO - dev);
    return v;
}

static int schoolSide(const Position &p, int c) {
    int w = kingWeakness(p, c), pr = attackPrep(p, c);
    return w + pr + pawnIntel(p, c) + sacComp(p, c, w, pr) + transition(p, c);
}
static int schoolTerms(const Position &p) {
    int d = schoolSide(p, WHITE) - schoolSide(p, BLACK);
    return max(-SCHOOL_CAP, min(SCHOOL_CAP, d));
}


// =====================================================================================
// ATTACK RACE (ETA) - tombana ny isan'ny moves (an'ny tenany) ilaina hatramin'ny mate :
//   Future Detection (piece manakaiky ny king), Top Defense (defenders tsara indrindra + vidin'ny fanalana azy),
//   Pawn Shield (vidin'ny fanokafana), Escape squares, + 1 (move farany).
//   Aorian'izay : comparaison ETA roa  ->  CARE / LET IT GO (raceCare) ary bonus ho an'ny mpiady haingana (raceTerm).
// =====================================================================================
static bool RACE_ON = true;
static int RACE_W = 5;            // cp isaky ny move eo amin'ny elanelana ETA
static int RACE_CAP = 90;         // cp
static int RACE_ALERT = 8;        // attack_power farany ambany
static int RACE_MARGIN = 4;       // elanelana (moves) ahafahana LET IT GO
static const int ETA_NONE = 40;
struct Eta { int eta, committed, d01, elim, escapes; };

static Eta attackEta(const Position &p, int c) {
    int o = c ^ 1, eks = kingSq(p, o), kf = fileOf(eks), kr = rankOf(eks);
    U64 zone = KING_ATT[eks], kz = zone | BIT(eks), occNoK = p.all & ~BIT(eks);
    int ds[20], nd = 0, committed = 0;
    U64 attE = 0;
    U64 pw = p.bb[mk(c, PAWN)];
    while (pw) {
        int s = poplsb(pw);
        U64 a = PAWN_ATT[c][s];
        attE |= a;
        if (a & kz) { committed++; if (nd < 20) ds[nd++] = 0; }
    }
    for (int pt = KNIGHT; pt <= QUEEN; pt++) {
        U64 b = p.bb[mk(c, pt)];
        while (b) {
            int s = poplsb(b), d;
            U64 a = attacksFrom(pt, s, p.all, c);
            attE |= attacksFrom(pt, s, occNoK, c);
            if (a & kz) { d = 0; committed++; }
            else {
                int ch = distCheb(s, eks);
                switch (pt) {
                    case KNIGHT: d = max(1, (ch - 1) / 2); break;
                    case BISHOP: d = ch <= 3 ? 1 : 2; break;
                    case ROOK: d = (abs(fileOf(s) - kf) <= 1 || abs(rankOf(s) - kr) <= 1) ? 1 : 2; break;
                    default: d = 1;
                }
            }
            if (nd < 20) ds[nd++] = d;
        }
    }
    Eta r{ETA_NONE, committed, 0, 0, 0};
    if (nd < 2) return r;                                   // tsy misy future azo atao (tsy ampy ny mpanafika)
    sort(ds, ds + nd);
    int d01 = ds[0] + ds[1];
    // Top Possible Defenses : defenders tsara indrindra (3 voalohany) sy ny vidin'ny fanalana azy
    int dw[16], dc[16], nde = 0;
    for (int pt = KNIGHT; pt <= QUEEN; pt++) {
        U64 b = p.bb[mk(o, pt)];
        while (b) {
            int s = poplsb(b);
            if (!(attacksFrom(pt, s, p.all, o) & kz)) continue;
            int ma = popcnt(attackersTo(p, s, c)), pr = popcnt(attackersTo(p, s, o));
            if (nde < 16) { dw[nde] = ATTW[pt]; dc[nde] = ma > pr ? 1 : (ma > 0 ? 2 : 3); nde++; }
        }
    }
    int elim = 0;
    for (int k = 0, lim = min(3, nde); k < lim; k++) {
        int bi = 0;
        for (int i = 1; i < nde; i++) if (dw[i] > dw[bi]) bi = i;
        elim += dc[bi]; dw[bi] = -1;
    }
    // Pawn Shield : vidin'ny fanokafana azy
    int shc = 0;
    U64 sh = zone & p.bb[mk(o, PAWN)];
    while (sh) { int s = poplsb(sh); shc += (attE & BIT(s)) ? 1 : 2; }
    shc = min((shc + 1) / 2, 4);
    int esc = popcnt(zone & ~p.occ[o] & ~(attE | KING_ATT[kingSq(p, c)]));
    int eta = d01 + elim + shc + (esc + 1) / 2 + 1;
    r.eta = min(eta, ETA_NONE); r.d01 = d01; r.elim = elim + shc; r.escapes = esc;
    return r;
}

// Surveillance : tsy mivily ny mpanafika mba hitandrina afa-tsy rehefa tena atao ny attack nofantarina (committed)
static double raceCare(const Eta &mine, const Eta &theirs, double myAdj, double thAdj) {
    if (theirs.eta >= ETA_NONE) return 0.0;
    double gap = thAdj - myAdj, f = 1.0;
    if (mine.eta < ETA_NONE) {
        if (gap >= RACE_MARGIN) f = 0.0;                            // LET IT GO
        else if (gap >= RACE_MARGIN / 2.0) f = 0.3;
        else if (gap >= 0) f = 0.6;
        else f = theirs.committed >= 2 ? 1.0 : (theirs.committed == 1 ? 0.7 : 0.4);   // CARE
        if (theirs.eta <= 6 && gap < 8) f = max(f, 0.6);            // loza efa akaiky loatra
    }
    return f;
}
static int raceTerm(const Position &p, const Eta e[2]) {
    int best = min(e[WHITE].eta, e[BLACK].eta);
    if (best >= ETA_NONE) return 0;
    double adjW = e[WHITE].eta - (p.stm == WHITE ? 0.5 : 0.0), adjB = e[BLACK].eta - (p.stm == BLACK ? 0.5 : 0.0);
    double gap = max(-10.0, min(10.0, adjB - adjW));
    double urgency = (ETA_NONE - best) / (double)ETA_NONE;
    int v = (int)(RACE_W * gap * urgency);
    return max(-RACE_CAP, min(RACE_CAP, v));
}

static int evaluate(const Position &p) {
    int npmW = npmOf(p, WHITE), npmB = npmOf(p, BLACK);
    bool eg = (npmW + npmB) <= 2800;
    int score = p.score;
    score += positionalOf(p, eg, WHITE) - positionalOf(p, eg, BLACK);
    score += kingSafety(p, WHITE) - kingSafety(p, BLACK);
    int aw = attackPower(p, WHITE), ab = attackPower(p, BLACK);
    score += (aw - ab) / 2;
    if (eg) score += endgameTerms(p, npmW, npmB);
    if (OPENING_ON && isOpeningPos(p, npmW, npmB)) score += openingTerms(p);
    if (OPEN2_ON && npmW + npmB >= OPEN2_NPM) score += castleStructTerms(p);
    if (OPEN2_ON && ROOT_US >= 0 && npmW + npmB >= OPEN2_NPM) { int d = castleVsOpp(p, ROOT_US); score += ROOT_US == WHITE ? d : -d; }
    if (STORM_ON && !eg) score += stormTerms(p);
    if (SCHOOL_ON && npmW + npmB >= SCHOOL_NPM) score += schoolTerms(p);
    Eta E[2];
    if (RACE_ON && (aw >= RACE_ALERT || ab >= RACE_ALERT)) {
        E[WHITE] = attackEta(p, WHITE); E[BLACK] = attackEta(p, BLACK);
        score += raceTerm(p, E);
    }
    return p.stm == WHITE ? score : -score;
}

// =====================================================================================
// GHOST SEARCH GUIDE - ny ghost rehetra dia ao amin'ny SEARCH ihany (tsy misy ao amin'ny evaluate intsony).
// Ghost = ny piece-ko vs king mpanohitra raha tsy misy defense (ny piece mpanohitra tsy sakana).
//   - ghostThreat      : coverage ghost vs real (cost = efa mila sacrifice/open line), pattern checkmate, tempo
//   - CARE / LET IT GO : (Race ETA na decideFactor) mamaritra ny fiarovana : tsy ahena raha CARE, ahena raha LET IT GO
//   - ghostGain        : move manampy ny coverage ghost amin'ny king mpanohitra = aggressive (tsy ahena, voalohany)
//   - castleDenialGain : move mikendry ny lalan'ny castling mpanohitra (ex-opening ghost)
//   - sacrificeVerdict : Sacrifice Evaluation (opening school) -> mendrika (+1 depth) / tsy mendrika (ahena)
//   - forcing          : ghost tempo <= 3 ary move manampy ny coverage -> +1 depth
// =====================================================================================
static bool GHOST_ON = true;
static int GHOST_ALERT = 10;      // attack_power farany ambany hampiasana ghost
static int GHOST_ORDER = 6000;    // bonus ordering ambony indrindra ho an'ny quiet move ghost
struct Threat { int score; int tempo; int cost; };

static double patternBonus(const Position &p, int c, int eks) {
    int ksq = kingSq(p, c);
    int ds[5][10], cnt[5];
    for (int t = 0; t < 5; t++) {
        cnt[t] = 0;
        U64 b = p.bb[mk(c, t)];
        while (b && cnt[t] < 10) ds[t][cnt[t]++] = distCheb(poplsb(b), eks);
        sort(ds[t], ds[t] + cnt[t]);
    }
    int f = fileOf(eks), r = rankOf(eks);
    int edge = min(min(f, 7 - f), min(r, 7 - r));
    double edgeFactor = 1 + (3 - edge) * 0.25;
    struct Pat { int need[5]; double w; };
    static const Pat PATS[] = {
        {{0, 0, 0, 0, 1}, 1.2}, {{0, 0, 0, 1, 1}, 1.3}, {{0, 0, 1, 0, 1}, 1.1}, {{0, 1, 0, 0, 1}, 1.1},
        {{0, 0, 0, 1, 0}, 1.0}, {{0, 0, 0, 2, 0}, 1.3}, {{0, 1, 0, 1, 0}, 1.0}, {{0, 0, 1, 1, 0}, 1.0},
        {{0, 2, 0, 0, 0}, 0.8}, {{0, 1, 1, 0, 0}, 0.9}, {{0, 0, 2, 0, 0}, 0.8}, {{0, 1, 2, 0, 0}, 1.0},
        {{2, 0, 0, 0, 0}, 0.7}, {{1, 0, 0, 1, 0}, 1.0}, {{1, 1, 0, 0, 0}, 0.9}, {{1, 0, 0, 0, 1}, 1.1},
    };
    double best = 0;
    for (const Pat &pt : PATS) {
        bool ok = true;
        for (int t = 0; t < 5; t++) if (cnt[t] < pt.need[t]) { ok = false; break; }
        if (!ok) continue;
        double sum = distCheb(ksq, eks); int n = 1;
        for (int t = 0; t < 5; t++)
            for (int i = 0; i < pt.need[t]; i++) { sum += ds[t][i]; n++; }
        double val = max(0.0, 8 - sum / n) * 3 * pt.w * edgeFactor;
        if (val > best) best = val;
    }
    return best;
}

static Threat ghostThreat(const Position &p, int c) {
    int o = c ^ 1, eks = kingSq(p, o);
    U64 zone = KING_ATT[eks], attG = 0, attR = 0, gOcc = p.occ[c] | BIT(eks);
    for (int pt = PAWN; pt <= KING; pt++) {
        U64 b = p.bb[mk(c, pt)];
        while (b) {
            int s = poplsb(b);
            attG |= attacksFrom(pt, s, gOcc, c);
            attR |= attacksFrom(pt, s, p.all, c);
        }
    }
    int gcov = popcnt(zone & attG), rcov = popcnt(zone & attR);
    int blocked = popcnt(zone & p.occ[o] & ~attG);
    int freeSq = popcnt(zone) - gcov - blocked;
    int cost = max(0, gcov - rcov);
    double score = rcov * 12 + cost * 6 + patternBonus(p, c, eks);
    if (attG & BIT(eks)) score += 15;
    int tempo = freeSq <= 3 ? (freeSq <= 1 ? 2 : 3) : (freeSq <= 5 ? 5 : 9);
    return Threat{(int)score, tempo, cost};
}

// fallback raha tsy mampiasa Race : ohatrinona no ahalan'i "mine" miraharaha ny attack mpanohitra
static double decideFactor(const Threat &mine, const Threat &theirs, bool meToMove) {
    if (theirs.tempo >= 9 || theirs.score < 25) return 0.0;
    double myT = mine.tempo - (meToMove ? 0.5 : 0.0), thT = theirs.tempo - (meToMove ? 0.0 : 0.5);
    if (mine.tempo <= 4 && myT < thT && mine.score >= theirs.score * 0.8) return 0.3;
    if (theirs.tempo >= 5) return 0.6;
    if (theirs.cost >= 3 && theirs.tempo >= 3) return 0.5;
    return 1.0;
}

struct GhostCtx {
    bool on = false;
    Threat mine{0, 9, 0}, theirs{0, 9, 0};
    double care = 0.0;               // 0 = LET IT GO, 1 = CARE (mitandrina ny attack mpanohitra)
    U64 ezone = 0, mzone = 0, gOcc = 0;
};
static GhostCtx ghostContext(const Position &p) {
    GhostCtx g;
    int me = p.stm, o = me ^ 1;
    if (attackPower(p, me) < GHOST_ALERT && attackPower(p, o) < GHOST_ALERT) return g;
    g.on = true;
    g.mine = ghostThreat(p, me);
    g.theirs = ghostThreat(p, o);
    if (g.theirs.score < 25) g.care = 0.0;
    else if (RACE_ON) {
        Eta em = attackEta(p, me), et = attackEta(p, o);
        g.care = raceCare(em, et, em.eta - 0.5, (double)et.eta);
    } else g.care = decideFactor(g.mine, g.theirs, true);
    int eks = kingSq(p, o), mks = kingSq(p, me);
    g.ezone = KING_ATT[eks] | BIT(eks);
    g.mzone = KING_ATT[mks] | BIT(mks);
    g.gOcc = p.occ[me] | BIT(eks);
    return g;
}

// coverage ghost : piece mihetsika from->to ka manampy firy ny toerana ao amin'ny king zone mpanohitra (ny piece mpanohitra tsy sakana)
static inline int ghostGain(int pt, int from, int to, int me, const GhostCtx &g) {
    return popcnt(attacksFrom(pt, to, g.gOcc, me) & g.ezone) - popcnt(attacksFrom(pt, from, g.gOcc, me) & g.ezone);
}
static inline int defenseGain(const Position &p, int pt, int from, int to, int me, const GhostCtx &g) {
    return popcnt(attacksFrom(pt, to, p.all, me) & g.mzone) - popcnt(attacksFrom(pt, from, p.all, me) & g.mzone);
}
// FANAKANANA ny castling mpanohitra : ny toerana ho lalovan'ny king (ghost : ny piece-ko ihany no sakana)
static U64 castlePath(const Position &p, int o) {
    if (isCastled(p, o)) return 0;
    int br = o == WHITE ? 0 : 7;
    bool ks = o == WHITE ? (p.castling & 1) : (p.castling & 4), qs = o == WHITE ? (p.castling & 2) : (p.castling & 8);
    U64 m = 0;
    if (ks) m |= BIT(br * 8 + 5) | BIT(br * 8 + 6);
    if (qs) m |= BIT(br * 8 + 3) | BIT(br * 8 + 2);
    return m;
}
static inline int castleDenialGain(const Position &p, int pt, int from, int to, int me, U64 path) {
    return popcnt(attacksFrom(pt, to, p.occ[me], me) & path) - popcnt(attacksFrom(pt, from, p.occ[me], me) & path);
}

// SACRIFICE EVALUATION (search) : 0 = tsy sacrifice, +1 = mendrika (compensation ampy), -1 = tsy mendrika
static int sacrificeVerdict(const Position &p, Move m, int gg, const GhostCtx &g, int weak, int prep) {
    int me = p.stm, o = me ^ 1, from = mFrom(m), to = mTo(m), pt = ptOf(p.sq[from]);
    if (pt == KING || (m & (F_CASTLE | F_EP)) || mPromo(m)) return 0;
    int victim = p.sq[to] != NOPIECE ? VALS[ptOf(p.sq[to])] : 0;
    U64 oa = attackersTo(p, to, o);
    if (!oa) return 0;
    int oppMin = 20000;
    for (int t = PAWN; t <= QUEEN; t++) if (oa & p.bb[mk(o, t)]) { oppMin = VALS[t]; break; }
    U64 myDef = attackersTo(p, to, me) & ~BIT(from);
    int loss = 0;
    if (!myDef) loss = VALS[pt] - victim;
    else if (oppMin < VALS[pt]) loss = VALS[pt] - oppMin - victim;
    if (loss < 60) return 0;
    int comp = weak + prep + 10 * max(gg, 0);
    bool worth = comp >= 25 + loss / 5;
    if (loss >= 250 && g.mine.tempo > 5 && gg < 2) worth = false;
    return worth ? 1 : -1;
}

// ordering : quiet move ghost / castle-denial -> aloha
static void ghostOrder(const Position &p, MoveList &l, const GhostCtx &g, U64 cpath, Move ttMove) {
    int me = p.stm;
    for (int i = 0; i < l.n; i++) {
        Move m = l.m[i];
        if (m == ttMove || (m & (F_CAP | F_EP | F_CASTLE)) || mPromo(m)) continue;
        int from = mFrom(m), to = mTo(m), pt = ptOf(p.sq[from]);
        if (pt == KING) continue;
        int b = 0;
        if (g.on) b += 1500 * max(0, ghostGain(pt, from, to, me, g));
        if (cpath) b += 1500 * max(0, castleDenialGain(p, pt, from, to, me, cpath));
        if (b) l.sc[i] += min(GHOST_ORDER, b);
    }
}

// =====================================================================================
// SEARCH
// =====================================================================================
static const int MATE = 100000, INF = 1000000, MAXPLY = 100;
struct TTEntry { U64 key; Move move; int32_t score; int8_t depth; uint8_t flag; };
static vector<TTEntry> TT;
static size_t TTMASK = 0;
static size_t ttMB = 64;
static void ttResize(size_t mb) {
    ttMB = mb;
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
static long long statPawnSac = 0;                   // isan'ny extension pawn-sacrifice (hita ao amin'ny info string)
static vector<Move> rootBan;                        // move ho sivanina amin'ny root (Opening Guard)
static bool CLEAN_ON = true;                       // Clean Move : tsy ahena ny move mankany amin'ny attack, ahena ny retreat/passive
static bool GUIDE_LMR = true, GUIDE_EXT = true;   // ghost ho mpitarika ny search (tsy ny eval)
static int GUIDE_ALERT = 12;

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
    // JORO guide: raha mamely ny king mpanohitra ny ahy (attack context), ny move mitondra piece mankany
    // amin'ny king dia tsy ahena (LMR) ary ny sacrifice mankany amin'ny king dia lalinina (+1).
    bool atkCtx = false;
    U64 ezone = 0;
    int ekSq = 0;
    if ((GUIDE_LMR || GUIDE_EXT || CLEAN_ON) && depth >= 3 && !chk && attackPower(p, p.stm) >= GUIDE_ALERT) {
        atkCtx = true;
        int ek = kingSq(p, p.stm ^ 1);
        ekSq = ek;
        ezone = KING_ATT[ek] | BIT(ek);
    }
    // pawnCtx : attack efa misy, na rook/queen mifanandrify amin'ny king mpanohitra ankoatry ny sakana pawn (latent)
    bool pawnCtx = atkCtx;
    if (GUIDE_EXT && depth >= 3 && depth <= 8 && !chk) {
        if (!atkCtx) { ekSq = kingSq(p, p.stm ^ 1); ezone = KING_ATT[ekSq] | BIT(ekSq); }
        if (!pawnCtx) {
            U64 noPawns = p.all & ~(p.bb[mk(WHITE, PAWN)] | p.bb[mk(BLACK, PAWN)]);
            for (int q = ROOK; q <= QUEEN && !pawnCtx; q++) {
                U64 sl = p.bb[mk(p.stm, q)];
                while (sl) { int s2 = poplsb(sl); if (attacksFrom(q, s2, noPawns, p.stm) & ezone) { pawnCtx = true; break; } }
            }
        }
    }
    // GHOST (ao amin'ny search) : context ho an'ny side mihetsika
    GhostCtx gc;
    U64 cpath = 0;
    int sWeak = 0, sPrep = 0;
    if (GHOST_ON && depth >= 3 && !chk) {
        gc = ghostContext(p);
        if (isOpeningPos(p, npmOf(p, WHITE), npmOf(p, BLACK))) cpath = castlePath(p, p.stm ^ 1);
        if (gc.on) { sWeak = kingWeakness(p, p.stm); sPrep = attackPrep(p, p.stm); }
    }
    MoveList l; genMoves(p, l, false);
    scoreMoves(p, l, ttMove, ply);
    if (gc.on || cpath) ghostOrder(p, l, gc, cpath, ttMove);
    int best = -INF, origAlpha = alpha, legal = 0;
    Move bestMove = 0;
    Move quietsTried[64]; int nq = 0;
    for (int i = 0; i < l.n; i++) {
        pickMove(l, i);
        Move m = l.m[i];
        if (root && !rootBan.empty() && find(rootBan.begin(), rootBan.end(), m) != rootBan.end()) continue;
        bool quiet = !(m & (F_CAP | F_EP)) && !mPromo(m);
        bool aggr = false, sacr = false, retreat = false, pawnSac = false;
        int sacV = 0, defGain = 0;
        bool ghostFx = false;
        if (pawnCtx) {
            // PAWN SACRIFICE : pawn (capture na tafiky) izay manala ny sakana ka mampifandray rook/queen amin'ny king mpanohitra
            int from = mFrom(m), to = mTo(m), me = p.stm;
            if (ptOf(p.sq[from]) == PAWN && !mPromo(m)) {
                U64 occ2 = (p.all & ~BIT(from)) | BIT(to);
                if (m & F_EP) occ2 &= ~BIT(to + (me == WHITE ? -8 : 8));
                bool opens = false;
                for (int q = ROOK; q <= QUEEN && !opens; q++) {
                    U64 sl = p.bb[mk(me, q)];
                    while (sl) {
                        int s2 = poplsb(sl);
                        if (popcnt(attacksFrom(q, s2, occ2, me) & ezone) > popcnt(attacksFrom(q, s2, p.all, me) & ezone)) { opens = true; break; }
                    }
                }
                if (opens && ((m & F_CAP) || isAttacked(p, to, me ^ 1))) { pawnSac = true; aggr = true; }
            }
        }
        if (atkCtx) {
            int from = mFrom(m), to = mTo(m), pt = ptOf(p.sq[from]), me = p.stm;
            if (pt != KING) {
                int aTo = popcnt(attacksFrom(pt, to, p.all, me) & ezone), aFrom = popcnt(attacksFrom(pt, from, p.all, me) & ezone);
                aggr = aggr || aTo > aFrom;
                if ((m & F_CAP) && (ezone & BIT(to)) && p.sq[to] != NOPIECE && VALS[pt] > VALS[ptOf(p.sq[to])]) sacr = true;
                // Clean Move : Unnecessary Retreat = quiet move miala amin'ny king mpanohitra, tsy mitsoka loza
                if (CLEAN_ON && quiet && pt != PAWN && !aggr && distCheb(to, ekSq) > distCheb(from, ekSq) && !isAttacked(p, from, me ^ 1)) retreat = true;
            }
        }
        if (gc.on || cpath) {
            int from = mFrom(m), to = mTo(m), pt = ptOf(p.sq[from]), me = p.stm;
            if (pt != KING && !(m & F_CASTLE)) {
                int gg = gc.on ? ghostGain(pt, from, to, me, gc) : 0;
                int cdg = cpath ? castleDenialGain(p, pt, from, to, me, cpath) : 0;
                if (gg > 0 || cdg > 0) aggr = true;
                if (gc.on) {
                    sacV = sacrificeVerdict(p, m, gg, gc, sWeak, sPrep);
                    if (sacV > 0) aggr = true;
                    if (quiet) defGain = defenseGain(p, pt, from, to, me, gc);
                    ghostFx = quiet && gg > 0 && gc.mine.tempo <= 3 && gc.mine.score >= 40;
                }
            }
        }
        if (!makeMove(p, m)) continue;
        legal++;
        bool givesCheck = inCheck(p);
        int newDepth = depth - 1, v;
        if (GUIDE_EXT && sacr && ply < MAXPLY / 2) newDepth++;
        else if (GUIDE_EXT && pawnSac && ply < MAXPLY / 2 && depth <= 8) { newDepth++; statPawnSac++; }
        else if (GUIDE_EXT && sacV > 0 && ply < MAXPLY / 2 && depth <= 8) newDepth++;        // sacrifice mendrika
        else if (GUIDE_EXT && ghostFx && ply < MAXPLY / 2 && depth <= 6) newDepth++;          // forcing (ghost tempo <= 3)
        if (legal == 1) v = -search(p, newDepth, -beta, -alpha, ply + 1, true);
        else {
            int red = 0;
            if (depth >= 3 && (legal >= 4 || (retreat && legal >= 2)) && quiet && !chk && !givesCheck && !(GUIDE_LMR && aggr)) {
                red = 1 + (legal >= 8 && depth >= 5) + (legal >= 14 && depth >= 7) + (retreat ? 1 : 0);
            }
            if (gc.on && quiet && !chk && !givesCheck && depth >= 3) {
                if (red > 0 && defGain > 0 && gc.care >= 0.6) red = 0;                                   // CARE : tsy ahena ny fiarovana
                else if (!aggr && defGain > 0 && gc.care <= 0.0 && legal >= 6 && depth >= 4) red++;      // LET IT GO
                if (sacV < 0 && legal >= 3) red++;                                                       // sacrifice tsy mendrika
            }
            if (red > newDepth - 1) red = max(0, newDepth - 1);
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

// =====================================================================================
// OPENING GUARD - sivana amin'ny root (tsy misy vidiny ao anaty search). Mandrara ireo move quiet ao amin'ny opening izay:
//  (1) Passive Development : bishop mankany d3/e3 (d6/e6) eo anoloan'ny pawn d/e tsy mbola nihetsika
//  (2) Clean Move : move faharoa an'ny bishop/knight mankany amin'ny toerana azo tratrarina tamin'ny move 1 (na miverina
//      any amin'ny toerana nialany), raha tsy voatafika ilay piece, tsy capture, tsy check.
// =====================================================================================
static bool OPEN_GUARD = true;
static bool guardBlocks(const Position &p, Move m) {
    if (m & (F_CAP | F_EP | F_CASTLE)) return false;
    if (mPromo(m)) return false;
    int c = p.stm, from = mFrom(m), to = mTo(m), pt = ptOf(p.sq[from]);
    if (pt != BISHOP && pt != KNIGHT) return false;
    if (pt == BISHOP)
        for (int f = 3; f <= 4; f++)
            if (to == relSq(c, f, 2) && (p.bb[mk(c, PAWN)] & BIT(relSq(c, f, 1)))) return true;
    if (isAttacked(p, from, c ^ 1)) return false;
    for (int i = p.hply - 1; i >= 0; i--) {
        Move pm = p.hist[i].move;
        if (!pm || mTo(pm) != from) continue;
        if (p.hist[i].cap != NOPIECE) return false;
        int a = mFrom(pm);
        if (to == a) return true;
        if (pt == BISHOP && (attacksFrom(BISHOP, a, p.all & ~BIT(from), c) & BIT(to))) return true;
        return false;
    }
    return false;
}

// REPERTOIRE ONLY : tsy ekena ny move ivelan'ny repertoire JORO mandritra ny dingana voalohany (root ban).
//  White : 1.e4 ; 1.e4 e5 -> 2.f4 (King's Gambit) na 2.Nf3 ; 1.e4 e5 2.Nf3 Nc6 -> 3.Bc4 (Italian) na 3.d4 (Scotch)
//  Black : vs 1.e4 -> ...c5 (Sicilian) na ...e6 (French, ary ...d5) ; vs hafa -> ...Nf6 / ...g6 (King's Indian)
static bool REP_ONLY = true;
static bool repBans(const Position &p, Move m) {
    if (!REP_ONLY || p.hply > 6) return false;
    string ms = moveStr(m);
    auto h = [&](int i) { return moveStr(p.hist[i].move); };
    if (p.stm == WHITE) {
        if (p.hply == 0)
            return p.fullmove == 1 && p.bb[mk(WHITE, PAWN)] == RANK_BB[1] && undeveloped(p, WHITE) == 4 && ms != "e2e4";
        if (p.hply == 2 && h(0) == "e2e4" && h(1) == "e7e5") return ms != "f2f4" && ms != "g1f3";
        if (p.hply == 4 && h(0) == "e2e4" && h(1) == "e7e5" && h(2) == "g1f3" && h(3) == "b8c6") return ms != "f1c4" && ms != "d2d4";
        return false;
    }
    if (undeveloped(p, BLACK) != 4) return false;
    if (p.hply == 1) {
        if (h(0) == "e2e4") return ms != "c7c5" && ms != "e7e6";
        return ms != "g8f6" && ms != "g7g6";
    }
    if (p.hply == 3) {
        if (h(0) == "e2e4" && h(1) == "e7e6") return ms != "d7d5";
        if (h(0) != "e2e4") return ms != "g7g6" && ms != "f8g7" && ms != "g8f6";
    }
    return false;
}

static Move think(Position &p, const Limits &lim, int *outScore = nullptr, int *outDepth = nullptr) {
    nodes = 0; nodeLimit = lim.nodes; stopSearch = false; timeLimitMs = lim.movetime; statPawnSac = 0;
    ROOT_US = p.stm;
    tStart = chrono::steady_clock::now();
    memset(killers, 0, sizeof killers);
    memset(historyT, 0, sizeof historyT);
    MoveList root; genMoves(p, root, false);
    // misy move ara-dalàna ve?
    Move firstLegal = 0; int nLegal = 0;
    for (int i = 0; i < root.n; i++) if (makeMove(p, root.m[i])) { unmakeMove(p); if (!firstLegal) firstLegal = root.m[i]; nLegal++; }
    if (!nLegal) return 0;
    rootBan.clear();
    // REPERTOIRE ONLY : fotsy 1.e4 ; mainty Sicilian/French/KID (jereo repBans)
    if (OPEN_GUARD && p.hply <= 40 && isOpeningPos(p, npmOf(p, WHITE), npmOf(p, BLACK))) {
        for (int i = 0; i < root.n; i++) {
            Move m = root.m[i];
            if (repBans(p, m)) { rootBan.push_back(m); continue; }
            if (!guardBlocks(p, m) || !makeMove(p, m)) continue;
            bool gc = inCheck(p);
            unmakeMove(p);
            if (!gc) rootBan.push_back(m);
        }
        if ((int)rootBan.size() >= nLegal) rootBan.clear();
        if (!rootBan.empty()) {
            if (!quietOut) {
                printf("info string opening guard bans:");
                for (Move m : rootBan) printf(" %s", moveStr(m).c_str());
                printf("\n");
            }
            if (find(rootBan.begin(), rootBan.end(), firstLegal) != rootBan.end())
                for (int i = 0; i < root.n; i++)
                    if (find(rootBan.begin(), rootBan.end(), root.m[i]) == rootBan.end() && makeMove(p, root.m[i])) { unmakeMove(p); firstLegal = root.m[i]; break; }
        }
    }
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
        if (abs(score) > MATE - 200 && (!CLEAN_ON || depth >= MATE - abs(score))) break;   // Clean: hamarino ny mate fohy indrindra
        if (lim.movetime > 0 && elapsedMs() > lim.movetime * 0.6) break;   // tsy ho ampy ho an'ny depth manaraka
    }
    if (!quietOut && statPawnSac) { printf("info string pawn-sacrifice extensions %lld\n", statPawnSac); fflush(stdout); }
    if (outScore) *outScore = bestScore;
    if (outDepth) *outDepth = doneDepth;
    return best;
}

// ---- UCI
static Position gpos;
static void setPosition(const string &line) {
    skippedMoves = 0;
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
            while (is >> t) { Move m = findMoveStr(gpos, t); if (m && makeMove(gpos, m)) {} else skippedMoves++; }
        }
        return;
    }
    if (is >> tok && tok == "moves") {
        while (is >> tok) { Move m = findMoveStr(gpos, tok); if (m) makeMove(gpos, m); else skippedMoves++; }
    }
}

// =====================================================================================
// REPERTOIRE (opening book JORO) - ireto ihany no opening ampiasain'i JORO (tsy misy hafa) :
//   WHITE : King's Gambit (attaque mivantana), Italian Game (préparation sy attaque), Scotch Game (fanokafana centre)
//   BLACK : Sicilian Defense (counterattack), King's Indian Defense (King-side attack), French Defense (contre-jeu stratégique)
// Ny andalana dia UCI ; ny key (transposition) no ampiasaina.
// Ny White-lines dia mitahiry ny move White ihany ; ny Black-lines ny move Black ihany.
// Ny RepertoireOnly (repBans) no mitandrina ny voalohany : White 1.e4 ; Black = Sicilian/French vs 1.e4, KID vs hafa.
// Ivelan'ny book dia ny search + ny OPENING SCHOOL no mitarika (plans sy tactical patterns).
// =====================================================================================
static bool BOOK_ON = true;
static bool BOOK_RANDOM = true;
static int BOOK_PLIES = 24;          // farany ambany: tsy mampiasa book rehefa mihoatra ny ply ireo
struct BookEntry { U64 key; Move move; int weight; };
static vector<BookEntry> BOOK;
static Position bpos;                // position fiasana ho an'ny fananganana book

struct RepLine { int color; const char *name; const char *moves; };
static const RepLine REPERTOIRE[] = {
    // ---------------- WHITE : King's Gambit ----------------
    {WHITE, "King's Gambit: 3...g5 Bc4", "e2e4 e7e5 f2f4 e5f4 g1f3 g7g5 f1c4 g5g4 e1g1 g4f3 d1f3"},
    {WHITE, "King's Gambit: 3...d6 Bc4", "e2e4 e7e5 f2f4 e5f4 g1f3 d7d6 f1c4 g8f6 b1c3"},
    {WHITE, "King's Gambit: 3...Nf6", "e2e4 e7e5 f2f4 e5f4 g1f3 g8f6 e4e5 f6h5 f1e2"},
    {WHITE, "King's Gambit: 3...Be7", "e2e4 e7e5 f2f4 e5f4 g1f3 f8e7 f1c4 g8f6 e4e5"},
    {WHITE, "King's Gambit declined: 2...Bc5", "e2e4 e7e5 f2f4 f8c5 g1f3 d7d6 b1c3 g8f6 f1c4"},
    // ---------------- WHITE : Italian Game ----------------
    {WHITE, "Italian: Giuoco Piano", "e2e4 e7e5 g1f3 b8c6 f1c4 f8c5 c2c3 g8f6 d2d4 e5d4 c3d4 c5b4 c1d2 b4d2 b1d2 d7d5 e4d5 f6d5 e1g1"},
    {WHITE, "Italian: 4...d6", "e2e4 e7e5 g1f3 b8c6 f1c4 f8c5 c2c3 g8f6 d2d4 e5d4 c3d4 c5b4 b1c3 f6e4 e1g1"},
    {WHITE, "Italian: Two Knights Ng5", "e2e4 e7e5 g1f3 b8c6 f1c4 g8f6 f3g5 d7d5 e4d5 c6a5 c4b5 c7c6 d5c6 b7c6 b5e2"},
    {WHITE, "Italian: Evans Gambit", "e2e4 e7e5 g1f3 b8c6 f1c4 f8c5 b2b4 c5b4 c2c3 b4a5 d2d4 e5d4 e1g1"},
    // ---------------- WHITE : Scotch Game ----------------
    {WHITE, "Scotch: 4...Nf6", "e2e4 e7e5 g1f3 b8c6 d2d4 e5d4 f3d4 g8f6 d4c6 b7c6 e4e5 d8e7 d1e2 f6d5 c2c4"},
    {WHITE, "Scotch: 4...Bc5", "e2e4 e7e5 g1f3 b8c6 d2d4 e5d4 f3d4 f8c5 c1e3 d8f6 c2c3 g8e7 f1c4"},
    {WHITE, "Scotch: 4...Qh4", "e2e4 e7e5 g1f3 b8c6 d2d4 e5d4 f3d4 d8h4 b1c3 f8b4 f1e2 g8f6 d4c6 b7c6 e1g1"},
    {WHITE, "Scotch Gambit", "e2e4 e7e5 g1f3 b8c6 d2d4 e5d4 f1c4 f8c5 e1g1 d7d6 c2c3"},
    // ---------------- BLACK : Sicilian Defense ----------------
    {BLACK, "Sicilian vs 1.e4 (Najdorf)", "e2e4 c7c5 g1f3 d7d6 d2d4 c5d4 f3d4 g8f6 b1c3 a7a6 c1e3 e7e5 d4b3 c8e6"},
    {BLACK, "Sicilian: 6.Bg5", "e2e4 c7c5 g1f3 d7d6 d2d4 c5d4 f3d4 g8f6 b1c3 a7a6 c1g5 e7e6 f2f4 f8e7"},
    {BLACK, "Sicilian: 6.Be2", "e2e4 c7c5 g1f3 d7d6 d2d4 c5d4 f3d4 g8f6 b1c3 a7a6 f1e2 e7e5 d4b3 c8e6"},
    {BLACK, "Sicilian: 2.Nc3", "e2e4 c7c5 b1c3 b8c6 g2g3 g7g6 f1g2 f8g7 d2d3 d7d6"},
    {BLACK, "Sicilian: 2.c3", "e2e4 c7c5 c2c3 g8f6 e4e5 f6d5 d2d4 c5d4 g1f3 b8c6 c3d4 d7d6"},
    // ---------------- BLACK : King's Indian Defense ----------------
    {BLACK, "KID: Classical", "d2d4 g8f6 c2c4 g7g6 b1c3 f8g7 e2e4 d7d6 g1f3 e8g8 f1e2 e7e5 e1g1 b8c6 d4d5 c6e7"},
    {BLACK, "KID: Fianchetto", "d2d4 g8f6 c2c4 g7g6 g2g3 f8g7 f1g2 e8g8 b1c3 d7d6 g1f3 b8d7 e1g1 e7e5"},
    {BLACK, "KID: vs Nf3", "d2d4 g8f6 g1f3 g7g6 c2c4 f8g7 b1c3 e8g8 e2e4 d7d6 f1e2 e7e5 e1g1 b8c6"},
    {BLACK, "KID vs 1.c4", "c2c4 g8f6 b1c3 g7g6 e2e4 d7d6 d2d4 f8g7 g1f3 e8g8"},
    {BLACK, "KID vs 1.Nf3", "g1f3 g8f6 c2c4 g7g6 b1c3 f8g7 e2e4 d7d6 d2d4 e8g8"},
    // ---------------- BLACK : French Defense ----------------
    {BLACK, "French: Classical 4.e5", "e2e4 e7e6 d2d4 d7d5 b1c3 g8f6 e4e5 f6d7 f2f4 c7c5 g1f3 b8c6 c1e3 a7a6"},
    {BLACK, "French: Tarrasch", "e2e4 e7e6 d2d4 d7d5 b1d2 c7c5 e4d5 e6d5 g1f3 b8c6 f1b5 f8d6 d4c5 d6c5 e1g1 g8e7"},
    {BLACK, "French: Advance", "e2e4 e7e6 d2d4 d7d5 e4e5 c7c5 c2c3 b8c6 g1f3 d8b6 a2a3 c5c4"},
    {BLACK, "French: Exchange", "e2e4 e7e6 d2d4 d7d5 e4d5 e6d5 g1f3 g8f6 f1d3 f8d6 e1g1 e8g8"},
    {BLACK, "French: Winawer", "e2e4 e7e6 d2d4 d7d5 b1c3 f8b4 e4e5 c7c5 a2a3 b4c3 b2c3 g8e7"},
};

static inline U64 bookKey(const Position &p) { return p.ep >= 0 ? (p.key ^ ZEPF[fileOf(p.ep)]) : p.key; }
static void bookAdd(U64 key, Move m) {
    for (auto &b : BOOK) if (b.key == key && b.move == m) { b.weight++; return; }
    BOOK.push_back(BookEntry{key, m, 1});
}
static void buildBook() {
    BOOK.clear();
    for (const RepLine &L : REPERTOIRE) {
        setFen(bpos, STARTFEN);
        istringstream is(L.moves);
        string t;
        int ply = 0;
        while (is >> t) {
            Move m = findMoveStr(bpos, t);
            if (!m) { fprintf(stderr, "[book] andalana tsy ara-dalàna: '%s' ao amin'ny %s (ply %d)\n", t.c_str(), L.name, ply + 1); break; }
            if (bpos.stm == L.color) bookAdd(bookKey(bpos), m);
            makeMove(bpos, m);
            ply++;
        }
    }
}
static Move bookProbe(const Position &p) {
    if (!BOOK_ON || p.hply >= BOOK_PLIES || p.halfmove > 40) return 0;
    int total = 0;
    U64 bk = bookKey(p);
    for (auto &b : BOOK) if (b.key == bk) total += b.weight;
    if (!total) return 0;
    int pick = 0;
    if (BOOK_RANDOM) {
        int r = (int)(rnd64() % (U64)total), acc = 0;
        for (auto &b : BOOK) if (b.key == bk) { acc += b.weight; if (r < acc) { pick = b.move; break; } }
    } else {
        int bw = -1;
        for (auto &b : BOOK) if (b.key == bk && b.weight > bw) { bw = b.weight; pick = b.move; }
    }
    return (Move)pick;
}

// Book = fototra foana ; ny search dia manamarina fotsiny fa tsy diso mafy ilay move (BookMargin cp ambanin'ny tsara indrindra)
static bool BOOK_CHECK = true;
static int BOOK_MARGIN = 120;
static bool bookSafe(Position &p, Move bm) {
    if (!BOOK_CHECK) return true;
    bool sq = quietOut, sg = OPEN_GUARD;
    quietOut = true;
    Limits lim; lim.depth = 6;
    int sBest = 0, d = 0;
    Move best = think(p, lim, &sBest, &d);
    bool ok = true;
    if (best && best != bm) {
        OPEN_GUARD = false;
        if (makeMove(p, bm)) {
            Limits l2; l2.depth = 5;
            int sOpp = 0;
            Move r = think(p, l2, &sOpp, &d);
            unmakeMove(p);
            if (r && -sOpp < sBest - BOOK_MARGIN) ok = false;
        }
    }
    OPEN_GUARD = sg; quietOut = sq;
    return ok;
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
    Move bk = bookProbe(gpos);
    if (bk && !bookSafe(gpos, bk)) {
        printf("info string book move %s rejected by search check\n", moveStr(bk).c_str());
        bk = 0;
    }
    if (bk) {
        printf("info string book move %s\n", moveStr(bk).c_str());
        printf("bestmove %s\n", moveStr(bk).c_str());
        fflush(stdout);
        return;
    }
    int sc, dp;
    Move m = think(gpos, lim, &sc, &dp);
    printf("bestmove %s\n", m ? moveStr(m).c_str() : "0000");
    fflush(stdout);
}

// ---- SELF-PLAY A/B : A = Race+Clean+Opening2 ON ; B = OFF ; ny safidy hafa (Ghost-search, Opening, Guide...) dia mitovy ho an'ny roa
static Position spos;
static void setAdv(bool on) { RACE_ON = on; CLEAN_ON = on; OPEN2_ON = on; OPEN_GUARD = on; STORM_ON = on; SCHOOL_ON = on; }
static void selfplay(int games, int depth) {
    bool sR = RACE_ON, sC = CLEAN_ON, sO = OPEN2_ON, sG = OPEN_GUARD, sS = STORM_ON, sQ = quietOut, sSc = SCHOOL_ON;
    U64 sRng = rng_state;
    size_t sTT = ttMB;
    ttResize(16);
    quietOut = true;
    int win = 0, draw = 0, loss = 0;
    for (int g = 0; g < games; g++) {
        rng_state = 0xC0FFEE123ULL + (U64)(g / 2) * 0x9E3779B97F4A7C15ULL;       // opening mitovy ho an'ny tsiroaroa
        setFen(spos, STARTFEN);
        bool aWhite = (g % 2 == 0);
        for (int i = 0; i < 6; i++) {
            MoveList l; genMoves(spos, l, false);
            for (int tries = 0; l.n && tries < 60; tries++) if (makeMove(spos, l.m[rnd64() % l.n])) break;
        }
        int result = 0, hi[2] = {0, 0};
        for (int ply = 0; ply < 300; ply++) {
            if (isDrawPos(spos)) break;
            bool aMoves = (spos.stm == WHITE) == aWhite;
            setAdv(aMoves);
            fill(TT.begin(), TT.end(), TTEntry{0, 0, 0, 0, 0});
            Limits lim; lim.depth = depth;
            int sc = 0, dp = 0, side = spos.stm;
            Move m = think(spos, lim, &sc, &dp);
            if (!m) { if (inCheck(spos)) result = aMoves ? -1 : 1; break; }
            if (sc >= 900) hi[side]++; else hi[side] = 0;
            if (hi[side] >= 4) { result = aMoves ? 1 : -1; break; }                // adjudication
            makeMove(spos, m);
        }
        if (result > 0) win++; else if (result < 0) loss++; else draw++;
        printf("game %d (%s): %s\n", g + 1, aWhite ? "A=white" : "A=black", result > 0 ? "A wins" : (result < 0 ? "B wins" : "draw"));
        fflush(stdout);
    }
    double pct = games ? 100.0 * (win + 0.5 * draw) / games : 0;
    printf("selfplay depth %d: A(Race+Clean+Opening2+Guard) +%d =%d -%d  score %.1f%%\n", depth, win, draw, loss, pct);
    fflush(stdout);
    RACE_ON = sR; CLEAN_ON = sC; OPEN2_ON = sO; OPEN_GUARD = sG; STORM_ON = sS; SCHOOL_ON = sSc; quietOut = sQ; rng_state = sRng;
    ttResize(sTT);
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
    buildBook();
    setFen(gpos, STARTFEN);
    string line;
    while (getline(cin, line)) {
        if (line.empty()) continue;
        istringstream is(line);
        string cmd;
        is >> cmd;
        if (cmd == "uci") {
            printf("id name JORO-cpp 1.2\nid author JORO\noption name Hash type spin default 64 min 1 max 4096\noption name Opening type check default true\noption name OpenCap type spin default 100 min 0 max 400\noption name Ghost type check default true\noption name GhostAlert type spin default 10 min 0 max 200\noption name GhostOrder type spin default 6000 min 0 max 20000\noption name School type check default true\noption name SchoolCap type spin default 90 min 0 max 300\noption name RepertoireOnly type check default true\noption name Seed type spin default 0 min 0 max 2147483647\noption name GuideLMR type check default true\noption name GuideExt type check default true\noption name GuideAlert type spin default 12 min 0 max 200\noption name Race type check default true\noption name RaceW type spin default 5 min 0 max 100\noption name RaceCap type spin default 90 min 0 max 400\noption name RaceAlert type spin default 8 min 0 max 200\noption name RaceMargin type spin default 4 min 1 max 30\noption name Clean type check default true\noption name Opening2 type check default true\noption name Opening2Cap type spin default 60 min 0 max 300\noption name Storm type check default true\noption name StormCap type spin default 80 min 0 max 300\noption name SameSidePen type spin default 35 min 0 max 200\noption name BookCheck type check default true\noption name BookMargin type spin default 120 min 0 max 500\noption name Book type check default true\noption name OpeningGuard type check default true\noption name BookRandom type check default true\noption name BookPlies type spin default 24 min 0 max 80\nuciok\n");
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
            else if (name == "Ghost") GHOST_ON = (val == "true" || val == "1");
            else if (name == "GhostOrder") GHOST_ORDER = max(0, atoi(val.c_str()));
            else if (name == "School") SCHOOL_ON = (val == "true" || val == "1");
            else if (name == "SchoolCap") SCHOOL_CAP = max(0, atoi(val.c_str()));
            else if (name == "Seed") { long long sd = atoll(val.c_str()); if (sd > 0) rng_state = 0x9E3779B97F4A7C15ULL ^ ((U64)sd * 0xD1B54A32D192ED03ULL); if (!rng_state) rng_state = 1; }
            else if (name == "RepertoireOnly") REP_ONLY = (val == "true" || val == "1");
            else if (name == "GhostAlert") GHOST_ALERT = max(0, atoi(val.c_str()));
            else if (name == "GuideLMR") GUIDE_LMR = (val == "true" || val == "1");
            else if (name == "GuideExt") GUIDE_EXT = (val == "true" || val == "1");
            else if (name == "GuideAlert") GUIDE_ALERT = max(0, atoi(val.c_str()));
            else if (name == "Race") RACE_ON = (val == "true" || val == "1");
            else if (name == "RaceW") RACE_W = max(0, atoi(val.c_str()));
            else if (name == "RaceCap") RACE_CAP = max(0, atoi(val.c_str()));
            else if (name == "RaceAlert") RACE_ALERT = max(0, atoi(val.c_str()));
            else if (name == "RaceMargin") RACE_MARGIN = max(1, atoi(val.c_str()));
            else if (name == "Clean") CLEAN_ON = (val == "true" || val == "1");
            else if (name == "Opening2") OPEN2_ON = (val == "true" || val == "1");
            else if (name == "Opening2Cap") OPEN2_CAP = max(0, atoi(val.c_str()));
            else if (name == "OpeningGuard") OPEN_GUARD = (val == "true" || val == "1");
            else if (name == "Storm") STORM_ON = (val == "true" || val == "1");
            else if (name == "StormCap") STORM_CAP = max(0, atoi(val.c_str()));
            else if (name == "SameSidePen") SAMESIDE_PEN = max(0, atoi(val.c_str()));
            else if (name == "BookCheck") BOOK_CHECK = (val == "true" || val == "1");
            else if (name == "BookMargin") BOOK_MARGIN = max(0, atoi(val.c_str()));
            else if (name == "Book") BOOK_ON = (val == "true" || val == "1");
            else if (name == "BookRandom") BOOK_RANDOM = (val == "true" || val == "1");
            else if (name == "BookPlies") BOOK_PLIES = max(0, atoi(val.c_str()));
        } else if (cmd == "position") setPosition(line);
        else if (cmd == "go") goCommand(line);
        else if (cmd == "perft") {
            int d = 5; is >> d;
            auto t0 = chrono::steady_clock::now();
            U64 n = perft(gpos, d);
            double s = chrono::duration<double>(chrono::steady_clock::now() - t0).count();
            printf("perft %d = %llu (%.2fs)\n", d, (unsigned long long)n, s); fflush(stdout);
        } else if (cmd == "race") {
            Eta e[2] = {attackEta(gpos, WHITE), attackEta(gpos, BLACK)};
            const char *nm[2] = {"White", "Black"};
            for (int c = 0; c < 2; c++)
                printf("%s attack: eta %d (committed %d, move-to-reach %d, defense-cost %d, escapes %d)\n", nm[c], e[c].eta, e[c].committed, e[c].d01, e[c].elim, e[c].escapes);
            double a0 = e[0].eta - (gpos.stm == WHITE ? 0.5 : 0.0), a1 = e[1].eta - (gpos.stm == BLACK ? 0.5 : 0.0);
            printf("care White %.2f  care Black %.2f  raceTerm(white pov) %d\n", raceCare(e[0], e[1], a0, a1), raceCare(e[1], e[0], a1, a0), raceTerm(gpos, e));
            fflush(stdout);
        } else if (cmd == "opening") {
            static const char *cn[4] = {"none", "short", "long", "center"};
            const char *nm[2] = {"White", "Black"};
            for (int c = 0; c < 2; c++)
                printf("%s: castle %s, shield short %d long %d, passiveDev %d, castleStruct %d | commit %d vsOpp %d storm %d\n", nm[c], cn[predictCastle(gpos, c)],
                       shieldQuality(gpos, c, 0), shieldQuality(gpos, c, 1), passiveDev(gpos, c), castleStructSide(gpos, c),
                       castleCommit(gpos, c), castleVsOpp(gpos, c), stormSide(gpos, c));
            fflush(stdout);
        } else if (cmd == "fen") {
            printf("fen %s\nskipped %d\n", fenStr(gpos).c_str(), skippedMoves); fflush(stdout);
        } else if (cmd == "school") {
            const char *nm[2] = {"White", "Black"};
            for (int c = 0; c < 2; c++) {
                int w = kingWeakness(gpos, c), pr = attackPrep(gpos, c);
                printf("%s: kingWeakness(enemy) %d, attackPrep %d, pawnIntel %d, sacComp %d, transition %d | total %d\n", nm[c], w, pr,
                       pawnIntel(gpos, c), sacComp(gpos, c, w, pr), transition(gpos, c), schoolSide(gpos, c));
            }
            printf("schoolTerms(white pov) %d\n", schoolTerms(gpos));
            fflush(stdout);
        } else if (cmd == "ghost") {
            GhostCtx g = ghostContext(gpos);
            printf("ghost %s | mine: score %d tempo %d cost %d | theirs: score %d tempo %d cost %d | care %.2f\n", g.on ? "ON" : "off",
                   g.mine.score, g.mine.tempo, g.mine.cost, g.theirs.score, g.theirs.tempo, g.theirs.cost, g.care);
            fflush(stdout);
        } else if (cmd == "selfplay") {
            int g = 10, d = 5; is >> g >> d;
            selfplay(g, d);
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
