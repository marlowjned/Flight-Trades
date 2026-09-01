% input in feet
rocket_length = 8.5;

% convert every table to an arrray and put in 3d array [row, column, rocket length]
S = zeros(7499,15,13);
S(:,:,1) = table2array(readtable("Spaceshot_8ft.CSV"));
S(:,:,2) = table2array(readtable("Spaceshot_9ft.CSV"));
S(:,:,3) = table2array(readtable("Spaceshot_10ft.CSV"));
S(:,:,4) = table2array(readtable("Spaceshot_11ft.CSV"));
S(:,:,5) = table2array(readtable("Spaceshot_12ft.CSV"));
S(:,:,6) = table2array(readtable("Spaceshot_13ft.CSV"));
S(:,:,7) = table2array(readtable("Spaceshot_14ft.CSV"));
S(:,:,8) = table2array(readtable("Spaceshot_15ft.CSV"));
S(:,:,9) = table2array(readtable("Spaceshot_16ft.CSV"));
S(:,:,10) = table2array(readtable("Spaceshot_17ft.CSV"));
S(:,:,11) = table2array(readtable("Spaceshot_18ft.CSV"));
S(:,:,12) = table2array(readtable("Spaceshot_19ft.CSV"));
S(:,:,13) = table2array(readtable("Spaceshot_20ft.CSV"));

% reshape to 2d so that each row represents a full table of a rocket length (usable with interp1)
[rows, cols, pages] = size(S);
S_flattened = reshape(permute(S,[3,1,2]),pages,[]);

% interpolate
results_flattened = interp1(1:13,S_flattened,rocket_length-7,"linear");

% convert output to a table
result_array = reshape(results_flattened,rows,cols);
result_table = array2table(result_array,VariableNames=readtable("Spaceshot_8ft.CSV").Properties.VariableNames);