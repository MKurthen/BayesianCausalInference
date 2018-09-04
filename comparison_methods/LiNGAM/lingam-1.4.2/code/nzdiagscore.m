function s = nzdiagscore( W )

fmt = [repmat('%4d ', 1, size(W,2)-1), '%4d\n'];
fprintf(fmt, W.');   %transpose is important!
s = sum(1./diag(abs(W)));
    
