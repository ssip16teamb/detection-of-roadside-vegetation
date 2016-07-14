function result = Gabors_greendilate(img)

	[height,width, ~] = size(img);
    	padimg = zeros(height+2, width+2);
    	padimg(2:height+1,2:width+1) = img(:,:,2);
	result = padimg;
    
	for i=2:height-1
		for j=2:width-1
            
            diff = max([img(i-1,j-1,1) img(i-1,j-1,2) img(i-1,j-1,3)]) - min([img(i-1,j-1,1) img(i-1,j-1,2) img(i-1,j-1,3)]);
            if diff > 30
                result(i,j) = max(max(padimg(i-1:i+1,j-1:j+1)));
            end
		end
	end
		
	result = result(2:height+1, 2:width+1);
    
    result = uint8(result);
		
end