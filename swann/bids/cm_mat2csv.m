function cm_mat2csv(behf)
    disp(behf);
    try
        load(behf, 'task_timing');

        blocks = Block(1).trials;
        for b = 2:length(Block)
            blocks = [blocks, Block(b).trials];
        end
        struct2tsv(behf(1: end - 4), good_times);
    catch
        load(behf, 'trials');
        export(trials, 'File', strcat(behf(1: end - 4), '.tsv'), ...
               'Delimiter', '\t');
    end
end