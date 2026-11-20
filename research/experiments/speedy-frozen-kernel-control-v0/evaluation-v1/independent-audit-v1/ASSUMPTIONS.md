# Assumptions fixed before implementation

1. The producer's final schema is the authoritative record layout; the audit
   reimplements calculations and gate rules from the frozen contract.
2. The six root pins bind actual records; CONFIG is untrusted until its complete
   protected table has been hashed. Old scientific artifacts stay read-only.
3. Source JPEG decoding uses Pillow RGB uint8 / float32(255), matching the saved
   GT representation. Independent metrics promote to float64 before arithmetic.
4. Exact pixel identity means raw float32 bits, distinct from numeric equality.
   Numeric mismatch counts and bit identity are reported separately.
5. A zero GT MSE has explicit null/infinite-PSNR representation. No invented
   finite PSNR permits a missing signed-difference gate to pass.
6. Model identities and RPC records are pinned declarations. CPU artifact checks
   cannot independently prove historical GPU event boundaries or no overlap.
7. Root supplies the true host main record at evaluation-v1/EXTERNAL_RUN.json;
   internal controller time is insufficient. Root measures the final audit wall.
8. Static/help/tiny synthetic checks never create run-v1 or read real arrays.
