function save_dataset_splits(E, scenario_choice, saveRoot, E_opts, opts)
%SAVE_DATASET_SPLITS
% Save train/valid/test splits for the paper setting:
%   train = 30000
%   valid = 10000
%   test  = 5000
%
% Memory-friendly behavior:
% - train / valid: Ht only (returnHorg = false)
% - test         : Ht + Horg (returnHorg = true)
%
% Fixed split seeds:
% - train: 12345
% - valid: 78901
% - test : 219876
%
% Inputs:
%   E              : environment index for load_environment
%   scenario_choice: scenario index
%   saveRoot       : directory to save .mat files
%   E_opts         : struct passed to load_environment
%   opts           : optional control struct
%
% opts fields:
%   .Ntrain        (default 30000)
%   .Nvalid        (default 10000)
%   .Ntest         (default 5000)
%   .skipIfExists  (default true)
%   .saveMeta      (default true)
%   .verbose       (default true)
%
% Example:
%   E_opts = struct('Nsub',624,'Ntap',32,'Nr',1);
%   save_dataset_splits(4, 1, 'dataset_splits', E_opts);

    if nargin < 4 || isempty(E_opts)
        E_opts = struct();
    end
    if nargin < 5 || isempty(opts)
        opts = struct();
    end

    % -------------------------
    % Defaults
    % -------------------------
    if ~isfield(opts, 'Ntrain') || isempty(opts.Ntrain)
        opts.Ntrain = 30000;
    end
    if ~isfield(opts, 'Nvalid') || isempty(opts.Nvalid)
        opts.Nvalid = 10000;
    end
    if ~isfield(opts, 'Ntest') || isempty(opts.Ntest)
        opts.Ntest = 5000;
    end
    if ~isfield(opts, 'skipIfExists') || isempty(opts.skipIfExists)
        opts.skipIfExists = true;
    end
    if ~isfield(opts, 'saveMeta') || isempty(opts.saveMeta)
        opts.saveMeta = true;
    end
    if ~isfield(opts, 'verbose') || isempty(opts.verbose)
        opts.verbose = true;
    end

    % Fixed split seeds
    seed_train = 12345;
    seed_valid = 78901;
    seed_test  = 219876;

    if ~exist(saveRoot, 'dir')
        mkdir(saveRoot);
    end

    scenario_name = get_scenario_name(E, scenario_choice);

    trainFile = fullfile(saveRoot, sprintf('train_%s.mat', scenario_name));
    validFile = fullfile(saveRoot, sprintf('valid_%s.mat', scenario_name));
    testFile  = fullfile(saveRoot, sprintf('test_%s.mat',  scenario_name));

    if opts.verbose
        fprintf('=== save_dataset_splits ===\n');
        fprintf('E=%d, scenario=%s\n', E, scenario_name);
        fprintf('saveRoot=%s\n', saveRoot);
        fprintf('Seeds: train=%d, valid=%d, test=%d\n', ...
            seed_train, seed_valid, seed_test);
    end

    % -------------------------
    % Metadata
    % -------------------------
    meta = struct();
    meta.E = E;
    meta.scenario_choice = scenario_choice;
    meta.scenario_name = scenario_name;
    meta.Ntrain = opts.Ntrain;
    meta.Nvalid = opts.Nvalid;
    meta.Ntest = opts.Ntest;
    meta.seed_train = seed_train;
    meta.seed_valid = seed_valid;
    meta.seed_test  = seed_test;
    meta.E_opts = E_opts;

    % -------------------------
    % Train split: Ht only
    % -------------------------
    if ~(opts.skipIfExists && isfile(trainFile))
        if opts.verbose
            fprintf('[train] Generating %d samples...\n', opts.Ntrain);
        end

        E_opts_train = E_opts;
        E_opts_train.returnHorg = false;
        E_opts_train.seed = seed_train;

        [Ht_train, ~] = load_environment(E, opts.Ntrain, scenario_choice, E_opts_train);

        if opts.verbose
            fprintf('[train] Saving: %s\n', trainFile);
        end

        if opts.saveMeta
            save(trainFile, 'Ht_train', 'meta', '-v7.3');
        else
            save(trainFile, 'Ht_train', '-v7.3');
        end

        clear Ht_train;
    else
        if opts.verbose
            fprintf('[train] Skip existing file: %s\n', trainFile);
        end
    end

    % -------------------------
    % Valid split: Ht only
    % -------------------------
    if ~(opts.skipIfExists && isfile(validFile))
        if opts.verbose
            fprintf('[valid] Generating %d samples...\n', opts.Nvalid);
        end

        E_opts_valid = E_opts;
        E_opts_valid.returnHorg = false;
        E_opts_valid.seed = seed_valid;

        [Ht_valid, ~] = load_environment(E, opts.Nvalid, scenario_choice, E_opts_valid);

        if opts.verbose
            fprintf('[valid] Saving: %s\n', validFile);
        end

        if opts.saveMeta
            save(validFile, 'Ht_valid', 'meta', '-v7.3');
        else
            save(validFile, 'Ht_valid', '-v7.3');
        end

        clear Ht_valid;
    else
        if opts.verbose
            fprintf('[valid] Skip existing file: %s\n', validFile);
        end
    end

    % -------------------------
    % Test split: Ht + Horg
    % -------------------------
    if ~(opts.skipIfExists && isfile(testFile))
        if opts.verbose
            fprintf('[test] Generating %d samples...\n', opts.Ntest);
        end

        E_opts_test = E_opts;
        E_opts_test.returnHorg = true;
        E_opts_test.seed = seed_test;

        [Ht_test, Horg_test] = load_environment(E, opts.Ntest, scenario_choice, E_opts_test);

        if opts.verbose
            fprintf('[test] Saving: %s\n', testFile);
        end

        if opts.saveMeta
            save(testFile, 'Ht_test', 'Horg_test', 'meta', '-v7.3');
        else
            save(testFile, 'Ht_test', 'Horg_test', '-v7.3');
        end

        clear Ht_test Horg_test;
    else
        if opts.verbose
            fprintf('[test] Skip existing file: %s\n', testFile);
        end
    end

    if opts.verbose
        fprintf('Done.\n');
    end
end


function scenario_name = get_scenario_name(E, scenario_choice)
% Helper for readable filenames

    switch E
        case 1
            scenarios = { ...
                'spot1_3p5G_bus', ...
                'spot1_3p5G_no_bus', ...
                'spot2_3p5G_no_bus', ...
                'spot3_3p5G_no_bus' ...
            };
        case 2
            scenarios = { ...
                'CDL-A', ...
                'CDL-B', ...
                'CDL-C', ...
                'CDL-D', ...
                'CDL-E' ...
            };
        case 3
            scenarios = { ...
                'indoor', ...
                'outdoor' ...
            };
        case 4
            scenarios = { ...
                'Indoor_CloselySpacedUser_2_6GHz', ...
                'IndoorHall_5GHz', ...
                'SemiUrban_CloselySpacedUser_2_6GHz', ...
                'SemiUrban_300MHz', ...
                'SemiUrban_VLA_2_6GHz' ...
            };
        case 5
            scenarios = { ...
                'random' ...
            };
        otherwise
            error('Unsupported E=%d', E);
    end

    if scenario_choice < 1 || scenario_choice > numel(scenarios)
        error('Invalid scenario_choice=%d for E=%d', scenario_choice, E);
    end

    scenario_name = scenarios{scenario_choice};

    % Safer filename text
    scenario_name = strrep(scenario_name, '.', 'p');
    scenario_name = strrep(scenario_name, ' ', '_');
    scenario_name = strrep(scenario_name, '/', '_');
end