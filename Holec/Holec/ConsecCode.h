double ConsecSclMlt(const double* A, const double* B, const int len);
void ConsecTranspose(const double* src, double* dst, const int w, const int h);
void ConsecMMult(double* src1, double* src2, double* dst, const int src1h, const int src1w, const int src2w);
void ConsecBlock1MMult(double* src1, double* src2, double* dst, const int src1h, const int src1w, const int src2w, const int bs = 500);
void ConsecBlock2MMult(double* src1, double* src2, double* dst, const int src1h, const int src1w, const int src2w, const int bs = 120);
void ConsecBlock3MMult(double* src1, double* src2, double* dst, const int src1h, const int src1w, const int src2w, const int bs = 120);
void ConsecHolec(double* A, int n);
void Consec1Holec(double* src, double* dst, int n);
void Consec2Holec(double* src, double* dst, int n);