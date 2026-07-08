---
up: "[[adjudication-method]]"
related: []
created: 2026-07-06T00:00:00
updates: []
in: bibner-werk
tags:
  - ner
  - iaa
  - adjudication
  - contested-tokens
---

# SR-08 Adjudication: Contested Tokens by Doc

Purpose and entry-writing workflow now live in `adjudication-plan.md` §2.5.1. Section numbers below are the citation format used when referencing an entry from a Doccano comment (e.g. "see contested-tokens §1.1").

## 1. 4FRUKZNYSPWOAH27M2SAZWCJPWTICOK7

**doc_id**: `6`

**Text**: "Ein Erbermli=||che vñ Erschröckliche newe || Zeittung/ von einẽ Fünff jerigen Kind || so im Stettlein Hohen Elbe/ vnter dem || Herrn Wilhelm Mirskowzky/ geboren || ist worden/ da Gott sein gerechtes Vr=||theil vber dises Kindt gesandt hat/ das || es vom Hellischen Fewer angezündet/|| ... der bœsen Welt zum ab=||schrecken/ vom Fluchen vnd || Füllerey.|| Jm Thon.|| Kompt her zu mir ... ||"

**Resolution**: `TITLE[0:363]`, `OTHER_TITLE[366:382]` ("Jm Thon" + incipit — gundi's split).

### 1.1 Contested token: "Jm Thon" → "Im Thon"

Archaic spelling for "Im Thon" ("in the tune [of]") — early-modern J/I orthography, not a surname (no SoR marker ties it to a person) — introducing "Kompt her zu mir ...", the incipit of the hymn this ballad is sung to. A real, specific tune reference (unlike content-vacuous labels such as "Ein ander Gesang.", resolved `O`) gets `OTHER_TITLE`: it cites another existing work as a performance instruction, not a second title cataloged in this record.

## 2. AL3HZXKS4X7EQDNJFNIQHK2R3Y6BC7GU

**doc_id**: `44`

**Text**: "Sainte-Pallaye, Jean Baptiste de LaCurne de: Das Ritterwesen des Mittelalters nach seiner politischen und militärischen Verfassung / Aus dem Französischen des Herrn de la Curne de Sainte-Palaye mit Anmerkungen, Zusätzen und Vorrede von Johann Ludwig Klüber. - Nürnberg : Grattenauer Bd. 3. - 1791"

**Resolution**: pending — `PERSON[0:43]`, `TITLE[45:130]` agreed; `[236:256]` ("Johann Ludwig Klüber") disputed between `PERSON` (frankp/LLM) and `TRANSLATOR` (maria).

### 2.1 Contested token: "von Johann Ludwig Klüber" → PERSON or TRANSLATOR

"Aus dem Französischen ... mit Anmerkungen, Zusätzen und Vorrede von Johann Ludwig Klüber" explicitly credits Klüber with the notes/additions/preface, but no `übersetzt`-type verb names a translator — "Aus dem Französischen" alone might still imply Klüber did the translation too, the same bare construction already read as `TRANSLATOR` in Doc 262 ("Aus dem Englischen von J. Pezzl"). Open, not yet decided consistently: `adjudication-precedents.md` §2.

## 3. 5KPA3OZ2GY3LW24HA3J3QGD4TJ7QLI4X

**doc_id**: `116`

**Text**: "Das dritte Lied. Jm Thon: Wenn mein Stündlein vorhanden ist."

**Resolution**: `O[0:16]` ("Das dritte Lied."), `O[17:25]` ("Jm Thon:"), `TITLE[26:59]` ("Wenn mein Stündlein vorhanden ist").

### 3.1 Contested token: "Jm Thon" — citation vs. this record's own title

Unlike Doc 6, this record has no separate title of its own — "Das dritte Lied." is a content-vacuous ordinal label (names nothing, same failure as "Ein ander Gesang.", §1.1), and "Wenn mein Stündlein vorhanden ist" (a real, specific hymn incipit) is not citing another cataloged work but *is* this record's own conventional identifier. So it gets `TITLE`, not `OTHER_TITLE`. Auto-resolved at tier 2 (majority: frankp + maria merged the full string into one `TITLE`) and never reached adjudication — this correction has not yet been applied to the sheet. Open precedent: `adjudication-precedents.md` §2.

**⚠️ Data mismatch, unresolved (found 2026-07-08)**: `data/gold/gold_reconciled.jsonl`'s current label for doc 116 is `OTHER_TITLE[0:15]` + `TITLE[26:59]` — this matches *none* of: frankp/maria's raw majority (`TITLE[0:59]`, one merged span), gundi's dissent (`TITLE[0:24]` + `OTHER_TITLE[26:60]`), or this entry's own stated correct resolution above (`O[0:16]` + `O[17:25]` + `TITLE[26:59]`, i.e. the prefix fully excluded, not labeled `OTHER_TITLE`). Some correction was applied to the shipped gold data, but not the one this entry argues for. Needs PI review to determine which label is actually correct before this is used for training/eval.

## 4. MDBJSX3DFUNEEJ6MNAGSIVDOXV6IZ227

**doc_id**: `189`

**Text**: "Leich-Predigt : Bey Christlicher Beerdigung Der Weyland Wohl-Edlen/ Groß- Ehr und Tugendreichen Frauen Elisabeth Margarethen Struven/ Des ... Herrn Anthon Dollen/ Gräffl. Schaumburg-Lippischen Rahts und Schaumb. Land-Rentmeisters hinterlassener Witwen/ Welche Anno 1679. den 9. Septembris sehlig ... verschieden und den 15. eiusdem in ihr Ruh-Kämmerlein zum Stadthagen ist beygesetzet/ Gehalten"

**Resolution**: `O[0:13]` ("Leich-Predigt"), `TITLE[16:251]` ("Bey Christlicher Beerdigung ... hinterlassener Witwen"), `OTHER_TITLE[253:394]` ("Welche Anno 1679. ... beygesetzet/ Gehalten").

**Related docs** (same `contested:4` tag): doc 194 shares this shape — genre/lead-in excluded, `TITLE[0:129]` identifies the subject, `OTHER_TITLE[131:311]` covers the elaboration. Doc 191 shares the *tag* but not the resolution: its final label is a single unsplit `TITLE[0:628]` covering the whole 628-char string, no `O`-exclusion and no `OTHER_TITLE` split at all — because unlike this doc, it has no structural separator anywhere in the text, so the no-separator-present rule keeps everything merged (see §5.1's citation of this rule).

### 4.1 Contested token: "Leich-Predigt" — genre label, not TITLE

Same content-vacuous failure as "Das dritte Lied." (§3.1) and "Ein ander Gesang." (§1.1): the bare genre label names nothing specific to this record and gets `O`. The remainder splits in two, not one: `TITLE` covers only what identifies *who* the record is about — Elisabeth Margarethen Struven, named via her own identity and via her husband Anthon Dollen's name and office (`hinterlassener Witwen` depends grammatically on his office, so his credentials stay bound to his name, matching the guide's §5.8 credential-merge rule — Dollen does not get his own `PERSON` span; he's a relational identifier for the deceased, not a responsible party). `OTHER_TITLE` covers the death/burial narrative that follows — *when and where* she died, which elaborates on the occasion without identifying anyone new. Matches §5.7's principle that a subject's identifying life-dates stay in `TITLE`, but only the identifying part — narrative event detail is `OTHER_TITLE`, the obituary-headline-vs-body distinction.

## 5. DKHJSKUJZSSMORCO7GA2CDRZEHL6J3NC

**doc_id**: `198`

**Text**: "Trifolium Consolatorium, Oder Dreyfacher Hertzens-Trost Christlicher Eltern/ und anderer betrübten Seelen : Aus den Worten Pauli/ Rom. 14. vers. 7. 8. 9. Unser Keiner [et]c. In einer einfältigen Leichen-Predigt/ Als Des ... Pauli Höpners/ S.S. Theol. Licentiati, der Kirchen zu Leißnig ... Pastoris und Superintendentis einiges ... Söhnlein Johann Paulus Höpner/ Welches den 14. Novembr. dieses ... 1649 Jahres ... sein Leben selig beschlossen/ den folgenden 18 Nov. ... zur Ruhe gebracht worden ... mitgetheilet"

**Resolution**: `TITLE[0:105]` ("Trifolium Consolatorium ... Seelen"), `OTHER_TITLE[108:173]` ("Aus den Worten Pauli ... Unser Keiner [et]c."), `TITLE[174:361]` ("In einer einfältigen Leichen-Predigt ... Söhnlein Johann Paulus Höpner"), `OTHER_TITLE[363:512]` ("Welches den 14. Novembr. ... mitgetheilet").

**⚠️ Related docs** (same `contested:5` tag) — **flagged, not confirmed consistent**: docs 233 and 234 share this tag but their final labels put `OTHER_TITLE` *first* (`OTHER_TITLE[0:89]`/`TITLE[92:235]`/`OTHER_TITLE[237:343]` for 233; `OTHER_TITLE[0:173]`/`TITLE[202:382]`/`OTHER_TITLE[384:517]` for 234), the reverse of this doc's `TITLE`-first structure. Both open with a moralizing/biblical-figure phrase before the subject's name and office, which the annotators treated as elaboration (`OTHER_TITLE`) rather than the record's own identifying title (`TITLE`) — possibly a real, reasoned distinction from this doc's "Trifolium Consolatorium," or possibly an inconsistency across the group. Not resolved here; needs a PI look, not assumed to match.

### 5.1 Contested token: "Trifolium Consolatorium, Oder Dreyfacher Hertzens-Trost..." — this record's own TITLE, not a cited work

The test is independent existence outside this one printed item, not floweriness: a tune incipit (doc 6/116) is a documented, separately-existing hymn; "Trifolium Consolatorium..." has no corpus evidence of reuse (appears once in the 395-doc sample, unlike "Jm Thon," 2 occurrences, or the "Leich-Predigt" genre label, 9 occurrences) — an elaborate, formulaic-sounding Latin+German devotional title is standard for one-off Baroque occasional pieces, and floweriness alone doesn't prove it cites a separately-published book. So it gets `TITLE`, matching doc 270's "Inaestimabilis Thesauri depositum," not demoted like a citation. The biblical citation that follows splits off as its own `OTHER_TITLE`, matching how doc 270 separates its psalm citation from its treatise `TITLE`. "In einer einfältigen Leichen-Predigt" has no separator excluding it from what follows (per doc 191's no-separator rule), so it stays merged with "Als Des ... Söhnlein Johann Paulus Höpner" as the occasion-identifying `TITLE`. The trailing death/burial clause is a second `OTHER_TITLE` — same headline-vs-body distinction as doc 189 (§4.1) — since `OTHER_TITLE` appearing twice in one record already has precedent elsewhere in the corpus (doc 318). Two `TITLE` spans in one record departs from "one `TITLE` per record" (guide §3.1), same justified exception as doc 270 (§6.5 in the fiz version): the treatise title and the occasion headline are two structurally distinct things, not one elaborated title.

## 6. I7Y5EVXL6BYSPLF7YPHX7ZMIQZEU2XTR

**doc_id**: `270`

**Text**: "Inaestimabilis Thesauri depositum, Das ist/ Christliche und aus Gottes Wort geführte Leichpredigt/ uber den Spruch des 31 Psalms/ In deine Hände befehle ich meinen Geist/ [et]c. : Bey dem Begräbniß/ Der ... Frawen Margareten Volckmers/ Des ... Gregorii Hasenwinckels ... Pastoris zu Behrsell Hertzlieben Haußfrawen/ Welche Anno 1640. ... selig verschieden/ darauff Sontages hernach ... bestattet worden ... gehalten und auffgesetzet"

**Resolution**: `TITLE[0:33]` ("Inaestimabilis Thesauri depositum"), `OTHER_TITLE[119:169]` ("31 Psalms/ In deine Hände befehle ich meinen Geist"), `TITLE[180:314]` ("Bey dem Begräbniß/ ... Haußfrawen"), `OTHER_TITLE[316:432]` ("Welche Anno 1640. ... auffgesetzet"). The span `[33:119]` (", Das ist/ Christliche und aus Gottes Wort geführte Leichpredigt/ uber den Spruch des") is unlabeled (`O`). Overrides the Tier-2 majority auto-resolution (`TITLE[0:177]`, `OTHER_TITLE[180:432]`), which had merged the Latin title, the "Das ist" gloss, and the whole psalm citation into one `TITLE` span.

**Related docs** (same `contested:6` tag): docs 146 and 291 both match this exact shape — `TITLE`, unlabeled genre/citation lead-in, `OTHER_TITLE`, `TITLE`, `OTHER_TITLE` — confirmed against their final labels (146: `TITLE[0:20]`/`OTHER_TITLE[124:187]`/`TITLE[198:361]`/`OTHER_TITLE[363:489]`; 291: `TITLE[0:86]`/`OTHER_TITLE[103:186]`/`TITLE[188:361]`/`OTHER_TITLE[363:514]`).

### 6.1 Contested token: biblical incipit after "Das ist" — genre/lead-in is O, citation+incipit is one OTHER_TITLE span

"Das ist" is an `OTHER_TITLE` trigger (guide §3), not a language boundary — Latin-to-German glosses introduced by "Das ist" do not get `PARALLEL_TITLE`, which the design rationale (`sr08_label-design-rationale.md` §4) reserves for structurally-marked ` = `/` / ` restatements, explicitly excluding foreign-language subtitles. But not everything after "Das ist" is automatically `OTHER_TITLE`: "Christliche und aus Gottes Wort geführte Leichpredigt" is a generic genre descriptor ("a Christian sermon drawn from God's word") — content-vacuous in the same way as doc 189's bare "Leich-Predigt" (§4.1), and gets `O`. "uber den Spruch des" ("upon the verse of the") is a citation lead-in that names nothing on its own and stays `O` too, right up to the point a specific citation actually starts.

## 7. 5RRN6YTX6GDRXJEBR5HPGNDXQ24OFDOL

**doc_id**: `317`

**Text**: "Dreyfacher Schwedischer LorBeer Krantz vnd Triumphirende Sieges Krohne von Des Durchläuchtigsten Großmächt. Fürsten vnd Herrn Gustavi Adolphi ... Großmächtigen Heldenthaten, glücklichen expeditionen vnd ... glorwürdigsten Victorien, geflochte[n] vnd zugerichtet :[Aus war- und glaubhafften Historien, Vrkunden vnd Berichten mit sonderm Fleiß trewlich zusammen bracht]. 2, Josua Redivivus, Das ist, Dreyfachen Schwedischen LorBeer-Krantzes Vnd Triumphirender SiegsKrone Ander Theil : Von ... Gustav-Adolphi ... Königlichen Majestät Glorwirdigisten hohen Expeditionen ... von anfang des 1630. Jahres ... biß an die vor Leipzig ergangene ... Feldschlacht, vnd ... Victori, gethan vnd verrichtet"

**Resolution**: `TITLE[0:261]` ("Dreyfacher Schwedischer LorBeer Krantz ... geflochte[n] vnd zugerichtet"), `OTHER_TITLE[372:480]` ("Josua Redivivus, Das ist, ... Ander Theil"), `OTHER_TITLE[483:691]` ("Von ... Gustav-Adolphi ... gethan vnd verrichtet"). The bracketed cataloger's note and volume number `[261:372]` (" :[Aus war- und glaubhafften Historien...]. 2, ") are unlabeled (`O`).

### 7.1 Contested token: "Josua Redivivus..." — volume-specific subtitle, not a second TITLE

Collection-plus-volume structure, same pattern as docs 17/81/118: `TITLE` is the overall multi-volume chronicle's own title ("Dreyfacher Schwedischer LorBeer Krantz und Triumphirende Sieges Krohne," celebrating Gustav II Adolf's campaigns generally); the bracketed compiler's note and the volume number "2," are excluded per the shelfmark/volume-number precedent (`adjudication-precedents.md` §2); "Josua Redivivus, Das ist, ... Ander Theil" is *this volume's own* subtitle (naming it the "Second Part," casting Gustav Adolf as a latter-day Joshua) plus the volume's specific content range (1630 through the Battle of Breitenfeld near Leipzig) — both genuinely volume-specific, not content-vacuous, so both get `OTHER_TITLE` rather than a second `TITLE`, matching how docs 81/118 keep one `TITLE` per record. maria's dissenting merge of the whole 691-character string into one `TITLE` breaks with this established pattern rather than extending it.

The specific citation — the psalm number and its quoted incipit, "31 Psalms/ In deine Hände befehle ich meinen Geist" — is a real, specific reference and gets `OTHER_TITLE`, merged as **one** span rather than split at the internal "/". This matches doc 6's precedent (§1.1), where a tune name and its incipit were merged into a single `OTHER_TITLE` span rather than kept separate: a citation reference (tune name, or here, a psalm number) and the text it points to (an incipit) function together as one performance/textual instruction, not two independently meaningful units.

**Precedent for future bible-incipit cases**: genre labels and citation lead-in phrases ("Christliche ... Leichpredigt", "uber den Spruch des", "Aus dem Text", etc.) are `O`; the citation number/reference plus its quoted verse text are merged into one `OTHER_TITLE` span, not split.

## 8. XRILQDJPNKNA3SJZABIPCAJHBESEYDGC

**doc_id**: `20`

**related**: docs 118 (Topographia Galliae), 241 (Memoria Busschiana) — same "Latin, Oder German, no colon" shape, not written up separately; resolve the same way (one merged `TITLE`)

**Text**: "Sursum Deorsum! Oder Die alleredelste Sorgfalt Des weyland Durchlauchtigsten Fürsten und Herrn/ Herrn Johann Georgen Des Andern/ Hertzogens zu Sachsen/ Jülich/ Cleve und Berg ... : Aus Höchst-gedachter Ihrer Chur-Fürstl. Durchl. Symbolo und nach Anleitung der Worte Pauli/ Col. III,1. 2. Als Dero am 22. Augusti, Anno 1680. höchst-seeligst verblichener Leichnam ... Am 29. Eiusdem, war der XII. Sonntag nach Trinitatis, In die Kirche Auf dem Schloß Freudenstein zu Freyberg gesetzet worden"

Funeral sermon for Johann Georg II, Elector of Saxony (d. 1680). "Sursum Deorsum!" ("Upward, Downward!") is his own motto (`Symbolo`, explicitly named as such later in the string) — a Baroque wordplay pairing physical burial with the spiritual ascent of Colossians 3:1–2, cited explicitly in the text.

**Resolution**: `TITLE[0:174]` ("Sursum Deorsum! Oder Die alleredelste Sorgfalt ... Cleve und Berg ..."), `OTHER_TITLE[181:489]` ("Aus Höchst-gedachter ... gesetzet worden").

### 8.1 Contested token: "Sursum Deorsum! Oder ..." — content, not punctuation, decides this

Presence or absence of a colon isn't the actual test — pre-1700 titles don't follow ISBD punctuation conventions, so a colon here is not a designed structural signal the way it is in modern catalog records (guide §1: "pre-1700 titles place the author's credentials before the title with no separator" — punctuation in this era is typographic, not structural). The real question is content: can "Sursum Deorsum!" be read as a complete, distinct title on its own, cleanly separable from what follows? It can't, confidently — and "Die alleredelste Sorgfalt..." isn't a translation of it either, so there's no clean bilingual-pair boundary to split at. Given that uncertainty, the safer reading keeps the whole stretch — "Sursum Deorsum! Oder Die alleredelste Sorgfalt ... Johann Georgen Des Andern ... Cleve und Berg" — as one `TITLE`, same final answer as docs 118/241, but for content reasons, not because a colon happens to be absent.
