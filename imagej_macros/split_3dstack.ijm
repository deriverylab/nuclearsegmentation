path = getDirectory("Choose source Directory ");
path=replace(path, "\\\\","\\\\\\\\");
files = getFileList(path);
File.makeDirectory(path+"Sequences")
for (j=0; j<files.length; j++) {
 
	if (endsWith(path+files[j], ".tif")==1){
savename=path+"Sequences";
name=path+files[j];
name2=replace(name, ".tif","");
open(name);
print(name);
	
name=getInfo("image.filename");
dir=getDirectory("image");
print(name);
print(dir);
run("Duplicate...", "duplicate channels=1");
print(savename);
run("Image Sequence... ", "format=TIFF save="+savename);
close();
close();

}
}