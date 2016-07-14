img = imread('v9.jpg');
[width, height, ~] = size(img);

mask = zeros(width, height);
imgdil = img;

imgdil(:,:,2) = Gabors_greendilate(img);

mask(:,:) = (imgdil(:,:,2) - 10) > imgdil(:,:,1) & (imgdil(:,:,2) - 10) > imgdil(:,:,3);

mask = logical(mask);

mask = medfilt2(mask, [5, 5]);
mask = imopen(mask, strel('disk',5));
mask = imclose(mask, strel('disk',11));

imwrite(imgdil, 'v9_dilated.jpg');
imwrite(mask, 'v9_output.jpg');

%subplot(1,3,1), imshow(img);
%subplot(1,3,2), imshow(imgdil);
%subplot(1,3,3), imshow(mask);
