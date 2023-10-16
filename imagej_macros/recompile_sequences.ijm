//Takes a folder of sequences (up to 200 time points, can change in line 15) and outputs concatanated TIFs
path = getDirectory("Choose source Directory ");
path=replace(path, "\\\\","\\\\\\\\");
files = getFileList(path);
for (j=0; j<files.length; j++) {
	if (endsWith(path+files[j], "0000.tif")==1){
name=path+files[j];
open(name);
print(name);
name1=name;

core=replace(name, "0000.tif","");
print(core);

for(i=1; i<200; i++){
k = IJ.pad(i, 4);
if (File.exists(core + k + ".jpg")){
	namei=core + k + ".jpg";
	open(namei);
	run("Concatenate...", "open image1=name1 image2=namei");
	name1=getTitle();
	
	
	}
else{
	list = getList("image.titles");
	if (list.length==1){
		saveAs("TIFF", core + ".tif");
		close();
	}
}}}

else if (endsWith(path+files[j], "0000.jpg")==1){
name=path+files[j];
open(name);
print(name);
name1=name;

core=replace(name, "0000.jpg","");
print(core);

for(i=1; i<200; i++){
	k = IJ.pad(i, 4);
if (File.exists(core + k + ".jpg")){
	namei=core + k + ".jpg";
	open(namei);
	run("Concatenate...", "open image1=name1 image2=namei");
	name1=getTitle();
	
	
	}
else{
	list = getList("image.titles");
	if (list.length==1){
		saveAs("TIFF", core + ".tif");
		close();
	}}}}}