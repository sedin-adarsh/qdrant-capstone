# Test queries

These 5 queries are used by every experiment. Each one targets a different
20 Newsgroups topic. `embed.py` reads the numbered lines below, so keep the
format `N. query text` if you edit them.

1. a question about a graphics card driver
2. debate about gun control laws
3. space shuttle launch and NASA missions
4. hockey playoff predictions
5. encryption and government key escrow

| # | Query | Newsgroup it should match |
|---|-------|---------------------------|
| 0 | a question about a graphics card driver | comp.graphics / comp.os.ms-windows.misc / comp.sys.ibm.pc.hardware |
| 1 | debate about gun control laws | talk.politics.guns |
| 2 | space shuttle launch and NASA missions | sci.space |
| 3 | hockey playoff predictions | rec.sport.hockey |
| 4 | encryption and government key escrow | sci.crypt |

(The `#` column is the index used by `python compare.py --query N`.)
