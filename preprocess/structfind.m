function index=structfind(a,field,value)
% StructFind, Find the index of a certain string or value in a struct
%
%       index=structfind(a,field,value)
%
%  inputs,
%       a : A Matlab struct, for example a(1).name='red', a(2).name='blue';
%       field : The name of the field which is searched, for example 'name'
%       value : The search value, for example 'blue'
%
%  outputs,
%       index : The Struct index which match the search
%
%
% Example,
%
% a(1).name='blue';
% a(1).index=0;
% a(1).val='g';
%
% a(2).name='red';
% a(2).index=1;
% a(2).val=[1 0];
%
% a(3).name='green';
% a(3).index=2;
% a(3).val='g';
%
% a(4).name='black';
% a(4).index=3;
% a(4).val=[0 0 0];
%
% a(5).name='yellow';
% a(5).index=NaN;
% a(5).val=[0 1 1];
%
% a(6).name='orange';
% a(6).index=[];
% a(6).val=[1 1 0];
%
% a(7).name='brown';
% a(7).index=6;
% a(7).val={'12'};
%
% a(8).name='white';
% a(8).index=7;
% a(8).val.x=1;
%
% a(8).name='purple';
% a(8).index=8;
% a(8).val.child.value=2;
%
% index=structfind(a,'name','red');
% disp(['index : ' num2str(index)])
%
% index=structfind(a,'index',1);
% disp(['index : ' num2str(index)])
%
% index=structfind(a,'val',[0 0 0]);
% disp(['index : ' num2str(index)])
%
% index=structfind(a,'val','g');
% disp(['index : ' num2str(index)])
%
% index=structfind(a,'index',NaN);
% disp(['index : ' num2str(index)])
%
% index=structfind(a,'index',[]);
% disp(['index : ' num2str(index)])
%
% index=structfind(a,'val',{'12'});
% disp(['index : ' num2str(index)])
%
% index=structfind(a,'val.child.value',2);
% disp(['index : ' num2str(index)])
%
% Function is written by D.Kroon University of Twente (December 2010)

% We don't compare structs
if(isstruct(value)), 
    error('structfind:inputs','search value can not be a struct');
end

% Stop if field doesn't exist
if(~isfield(a,field))
    index=find(arrayfun(@(x)(cmp(x,field,value)),a,'uniformoutput',true));
else
    index=find(arrayfun(@(x)(cmp(x,field,value)),a,'uniformoutput',true));
end


function check=cmp(x,field,value)
check=false;
if(isfield(x,field))
    % Simple field like x.tag
    x=x.(field); 
else
    % Complex field like x.tag.child.value
    in=find(field=='.');
    s=[1 in+1]; e=[in-1 length(field)];
    for i=1:length(s)
        fieldt=field(s(i):e(i));
        if(isfield(x,fieldt)), x=x.(fieldt);  else return; end
    end
end

% We don't compare structs
if(isstruct(x)), return; end

% Values can only be equal, if they equal in length
if(length(x)==length(value)), 
    % This part compares the NaN values 
    if((~iscell(x))&&(~iscell(value))&&any(isnan(value))), 
        checkv=isnan(value); checkx=isnan(x);
        if(~all(checkx==checkv)), return; end
        x(checkx)=0; value(checkv)=0;
    end
    % This part compares for both string as numerical values 
    if(iscell(x)||iscell(value))
        check=all(strcmp(x,value)); 
    else
        check=all(x==value); 
    end
end
