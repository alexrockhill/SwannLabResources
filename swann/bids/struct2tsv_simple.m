function struct2tsv_simple(f, my_struct)
    fn = fieldnames(my_struct);
    tsv = '';
    for i=1: numel(fn)
        tsv = strcat(tsv, fn{i}, '\t');
    end
    tsv = strcat(tsv(1: strlength(tsv) - 2), '\n');
    
    for i=1:numel(fn)
        if isstruct(my_struct.(fn{i}))
            tsv = strcat(tsv, 'n/a\t');
            try
                struct2tsv(strcat(f, fn{i}), my_struct.(fn{i}));
            catch
                disp('Unable to convert:');
                disp(fn{i});
            end
        else
            k = numel(my_struct.(fn{i}));
            if k > 1
                for j=1:k
                    tsv = strcat(tsv, ...
                        char(num2str(my_struct.(fn{i})(j))), ', ');
                end
                tsv = tsv(1: strlength(tsv) - 1);
                tsv = strcat(tsv, '\t');
            else
                tsv = strcat(tsv, char(num2str(my_struct.(fn{i}))), '\t');
            end
        end
    end
    tsv = strcat(tsv(1: strlength(tsv) - 2), '\n');
    
    fileID = fopen(strcat(f, '.tsv'), 'w');
    fprintf(fileID, tsv);
    fclose(fileID);
end