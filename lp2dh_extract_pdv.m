function X = lp2dh_extract_pdv(video,P,indices)
%LP2DH_EXTRACT_PDV Historical P=3 neighbor-minus-center differences.

video = double(video);
[height,width,time] = size(video);
center = video(2:height-1,2:width-1,2:time-1);
valid_count = numel(center);
if nargin<3 || isempty(indices)
    indices = (1:valid_count)';
else
    indices = indices(:);
end

X = zeros(numel(indices),26);
column = 0;
for temporal = -1:1
    for horizontal = -1:1
        for vertical = -1:1
            if temporal==0 && horizontal==0 && vertical==0, continue; end
            column = column+1;
            neighbor = video((2:height-1)+vertical, ...
                (2:width-1)+horizontal,(2:time-1)+temporal);
            delta = neighbor-center;
            X(:,column) = delta(indices);
        end
    end
end
end
