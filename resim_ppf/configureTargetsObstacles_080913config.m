%%% File from amy regarding target locations %%%

targAngles       = nan(24,1);     periphRad = nan(24,1);          targRad_maj = nan(24,1);        targRad_min = nan(24,1);    targOrient = nan(24,1);
targAngles(1:12) = 0:30:(360-30); periphRad(1:12) = 6.5;          targRad_maj(1:12) = 1.2;        targRad_min(1:12) = 1.2;    targOrient(1:12) = 0;
                                                                  targRad_maj(2)    = 2;          targRad_min(2)    = 0.75;   targOrient(2)    = 30;
targAngles(13)   = 0;             periphRad(13)   = 6.5;          targRad_maj(13)   = 3.25;        targRad_min(13)   = 2.5;    targOrient(13)   = 0;         %large obstacles
targAngles(16)   = 180;           periphRad(16)   = 6.5;          targRad_maj(16)   = 3.25;        targRad_min(16)   = 2.5;    targOrient(16)   = 0;         %large obstacles
targAngles(20)   = 0;             periphRad(20)   = 0;            targRad_maj(20)   = 2;          targRad_min(20)   = 0.75;   targOrient(20)   = 90;        %small obstacles
targAngles(21)   = 0;             periphRad(21)   = 0;            targRad_maj(21)   = 2;          targRad_min(21)   = 0.75;   targOrient(21)   = 150;       %small obstacles
targAngles(23)   = 180;           periphRad(23)   = 4.25;         targRad_maj(23)   = 1.2;        targRad_min(23)   = 1.2;    targOrient(23)   = 0;         %short-reach, high-curvature targets in the middle
targAngles(24)   = 0;             periphRad(24)   = 4.25;         targRad_maj(24)   = 1.2;        targRad_min(24)   = 1.2;    targOrient(24)   = 0;

targAngles = np.zeros((24, ))
targAngles[:12] = np.arange(0, 361-30, 30)
targAngles[15] = 180;
targAngles[22] = 180;

periphRad = np.zeros((24, ))
periphRad[:12] = 6.5
periphRad[12] = 6.5
periphRad[15] = 6.5
periphRad[22] = 4.25
periphRad[23] = 4.25


targRad_maj = np.zeros((24, ))
targRad_min = np.zeros((24,))
targOrient = np.zeros((24, ))





targVectors = [cosd(targAngles) sind(targAngles)];

centerPos = [0 2.5];

targPos = repmat(periphRad, 1, 2).*targVectors + repmat(centerPos, size(targAngles));
% targPos(:,1) = targPos(:,1)-centerPos(1);
% targPos(:,2) = targPos(:,2)-centerPos(2);

%get target sizes etc. for plots
t           = linspace(0,360,100);
targObjects = nan(2, length(t), length(targAngles));
for i=1:length(targAngles)
    
    if ~isnan(targAngles(i))
        
        %targets with center at (0,0)
        targObjects(1,:,i) = targPos(i,1) + targRad_maj(i)*cosd(targOrient(i))*cosd(t) - targRad_min(i)*sind(targOrient(i))*sind(t) - centerPos(1);
        targObjects(2,:,i) = targPos(i,2) + targRad_maj(i)*sind(targOrient(i))*cosd(t) + targRad_min(i)*cosd(targOrient(i))*sind(t) - centerPos(2);
    end
end


%list of trial types
trialList(1,:)  = [5 2 11];   trialType(1)  = 1; %no curvature
trialList(2,:)  = [11 2 5];   trialType(2)  = 1; 
trialList(3,:)  = [3 13 11];  trialType(3)  = 2; % low curvature
trialList(4,:)  = [11 13 3];  trialType(4)  = 2;
trialList(5,:)  = [5 16 9];   trialType(5)  = 2;
trialList(6,:)  = [9 16 5];   trialType(6)  = 2;
trialList(7,:)  = [3 21 9];   trialType(7)  = 3;  %high curvature, long distance
trialList(8,:)  = [9 21 3];   trialType(8)  = 3;
trialList(9,:)  = [23 20 24]; trialType(9)  = 4; %high curvature, short distance
trialList(10,:) = [24 20 23]; trialType(10) = 4;

for j = 1:5
    figure; hold all
    for l = 1:2
        tl = ((j-1)*2)+l;
        for i = trialList(tl, :)
            plot(targObjects(1, :, i), targObjects(2, :, i));
        end
    end
end

%Add 63 to 'trialList' to get obstacle curves

