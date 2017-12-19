clear all
load combinedDB_20161220
combinedDB = combinedNew_5metrics_All;
eminDBscore = combinedDB.subjQual.MOS(1:100);
jpegDBscore = combinedDB.subjQual.MOS(101:240);
jpegXTDBscore = combinedDB.subjQual.MOS(241:480);
jpeg2kDBscore = combinedDB.subjQual.MOS(481:690);

eminDBscorevar = combinedDB.subjQual.Std(1:100);
jpegDBscorevar = combinedDB.subjQual.Std(101:240);
jpegXTDBscorevar = combinedDB.subjQual.Std(241:480);
jpeg2kDBscorevar = combinedDB.subjQual.Std(481:690);


% scores = [jpeg2kDBscore jpegDBscore jpegXTDBscore eminDBscore ];
% for ind = 1:length(dirSt)
lum = @(x) 0.2126*x(:,:,1) + 0.7152*x(:,:,2) + 0.0722*x(:,:,3);
fp = 0;

refid = 0;
oldref = 'none';

dirSt = dir('j2kDatabase\testHdr\*.exr');
for ind = 1:length(dirSt)
    fileName = dirSt(ind).name;
    idx = strfind(fileName,'_');
    tmoNames{ind+fp} = fileName(1:idx(1)-1);
    contentName = fileName(idx(1)+1:idx(2)-1);
    fileName = strcat('j2kDatabase\testHdr\',fileName);
    contentName = strcat('j2kDatabase\origHdr\',contentName,'.exr');
    fileNames{ind+fp} = fileName;
    contentNames{ind+fp} = contentName; 
    dtype(ind+fp) = 1; 
    ppa(ind+fp) =  hdrvdp_pix_per_deg(47,[1920 1080], 1.78);
    if ~strcmp(oldref,contentName)
        oldref = contentName;
        refid = refid + 1;
    end
    refidar( ind + fp ) = refid;
    {contentName refid}
end

fp = fp+ind;
dirSt = dir('jpegDatabase\test\*.exr');
for ind = 1:length(dirSt)
    fileName = dirSt(ind).name;
    idx = strfind(fileName,'_');
    tmoNames{ind+fp} = fileName(1:idx(1)-1);
    contentName = fileName(1:idx(1)-1);
    fileName = strcat('jpegDatabase\test\',fileName); 
    contentName = strcat('jpegDatabase\orig\',contentName,'.exr');
    fileNames{ind+fp} = fileName;
    contentNames{ind+fp} = contentName; 
    dtype(ind+fp) = 2;
    ppa(ind+fp) =  hdrvdp_pix_per_deg(47,[1920 1080], 1.78);
    if ~strcmp(oldref,contentName)
        oldref = contentName;
        refid = refid + 1;
    end
    refidar( ind + fp ) = refid;
    {contentName refid}
end

fp = fp+ind;
dirSt = dir('jpegXtDatabase_corrected\decoded\*.pfm');
% for ind = 1:length(dirSt)
lum = @(x) 0.2126*x(:,:,1) + 0.7152*x(:,:,2) + 0.0722*x(:,:,3);
for ind = 1:length(dirSt)
    fileName = dirSt(ind).name;
    idx = strfind(fileName,'_Pr');
    contentName = fileName(1:idx(1)-1);   
    fileName = strcat('jpegXtDatabase_corrected\decoded\',fileName);
    contentName = strcat('jpegXtDatabase_corrected\original\',contentName,'.pfm');
    fileNames{ind+fp} = fileName;
    contentNames{ind+fp} = contentName;     
    dtype(ind+fp) = 3;
    ppa(ind+fp) =  hdrvdp_pix_per_deg(47,[1920 1080], 1.92);
    if ~strcmp(oldref,contentName)
        oldref = contentName;
        refid = refid + 1;
    end
    refidar( ind + fp ) = refid;
    {contentName refid}
end



fp = fp+ind;

dirSt = dir('databaseStuff\databaseToPublish\Images\Luminance_Estimations_Test_Images\*.exr');
for ind = 1:length(dirSt)    
    
    itemName = dirSt(ind).name;
    fields = strsplit(itemName,'_');
    if strcmp ( fields{4},'jpeg')
         dtype(ind+fp) = 2;
    elseif strcmp( fields{4},'jp2k')
        dtype(ind+fp) = 1;
    elseif strcmp( fields{4},'jpxt' )
        dtype (ind+fp) = 3;
%     elseif strcmp (fields{4},'noComp')
%         dtype(ind+fp) = 4;
%         fields{4};
    else
        dtype(ind+fp) = 4;
        fields{4};
    end
    ppa(ind+fp) =  hdrvdp_pix_per_deg(47,[1920 1080], 1.20);
    orname = fields{2};
    orfName = strcat(['databaseStuff\databaseToPublish\Images\Luminance_Estimations_Originals\', orname, '.exr']);
    fileName = strcat(['databaseStuff\databaseToPublish\Images\Luminance_Estimations_Test_Images\', itemName]);
    fileNames{ind+fp} = fileName;
    contentNames{ind+fp} = orfName;
    
    if ~strcmp(orname,oldref)
        oldref = orname;
        refid = refid + 1;
    end
    refidar( ind + fp ) = refid;
    {orname refid}
end


[C, ia, ic] = unique(combinedDB.names.content);

emincon = ic(1:100);
jpegcon = ic(101:240);
jpegxtcon = ic(241:480);
jpeg2kcon = ic(481:690);

fileNames = fileNames';
contentNames = contentNames';
dtype = dtype';
scores = [jpeg2kDBscore; jpegDBscore; jpegXTDBscore; eminDBscore ];% combinedDB.subjQual.MOS;
refidar =  [jpeg2kcon; jpegcon; jpegxtcon;emincon ];
scvar = [jpeg2kDBscorevar; jpegDBscorevar; jpegXTDBscorevar;eminDBscorevar ];

fullFileName = 'Data\Reference';
if exist(fullFileName, 'file')
  pause(1)
else
  mkdir(fullFileName)  
end
fullFileName = 'Data\Distorted';
if exist(fullFileName, 'file')
  pause(1)
else
  mkdir(fullFileName) 
end


fileID = fopen('data.txt','w');
gen_images = 1;
dis_hist = 0;
for ind = 1:length(fileNames)
    try 
        
        fprintf(fileID,'%s,%s,%d,%d,%f\n',strcat( 'Data\Distorted\', num2str(ind), '.exr'),strcat( 'Data\Reference\', num2str(ind), '.exr'),refidar(ind),dtype(ind),scores(ind));
        if gen_images
            imdis = hdrimread( fileNames{ind} );
            imref = hdrimread( contentNames{ind});        
            hdrimwrite(imdis, strcat( 'Data\Distorted\', num2str(ind), '.exr'));
            hdrimwrite(imref, strcat( 'Data\Reference\', num2str(ind), '.exr'));
            str = sprintf('Read %s successfully. Writing file to %s.', fileNames{ind}, 'Data\Distorted\', num2str(ind), '.tiff');
            display(str)
            if dis_hist
                x = lum(imdis);
                hist(x(:))
                title(fileNames{ind})
                pause(1)
            end
        end
        
    catch ME
        display('Image Read Error. Cant read')        
        fileNames{ind}
        pause(10)
    end
end
fclose(fileID);
% 
% length(fileNames)
% save( 'FilenameAndScores.mat', 'fileNames','contentNames','dtype','scores','refidar', 'scvar','ppa');
