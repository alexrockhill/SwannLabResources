function slow_fast_mat2csv(behf)
    disp(behf);
    try
        load(behf, 'parameters');
        struct2tsv_simple(strcat(behf(1: end - 4), 'parameters'), parameters);
    catch
        disp('Unable to load parameters')
    end
    try
        load(behf, 'Block');

        blocks = Block(1).trials;
        for b = 2:length(Block)
            blocks = [blocks, Block(b).trials];
        end
        struct2tsv(behf(1: end - 4), blocks);
    catch
        load(behf, 'trials');
        export(trials, 'File', strcat(behf(1: end - 4), '.tsv'), ...
               'Delimiter', '\t');
    end
end