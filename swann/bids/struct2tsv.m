function struct2tsv(f, my_struct)
    fn = fieldnames(my_struct);
    tsv = '';
    for i=1: numel(fn)
        tsv = strcat(tsv, fn{i}, '\t');
    end
    tsv = strcat(tsv(1: strlength(tsv) - 2), '\n');
    
    for j = 1: length(my_struct)
        for i=1:numel(fn)
            if isempty(my_struct(j).(fn{i}))
                tsv = strcat(tsv, 'n/a\t');
            else
                tsv = strcat(tsv, char(num2str(my_struct(j).(fn{i}))), ...
                             '\t');
            end
        end
        tsv = strcat(tsv(1: strlength(tsv) - 2), '\n');
    end
    tsv = strcat(tsv(1: strlength(tsv) - 2), '\n');
    
    fileID = fopen(strcat(f, '.tsv'), 'w');
    fprintf(fileID, tsv);
    fclose(fileID);
end