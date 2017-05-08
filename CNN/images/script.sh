# a=517
# for i in *.jpeg;
# do
# 	new=$(printf "%d.jpeg" "$a"); 
# 	mv -- "$i" "$new";
# 	let a=a+1; 
# done
a=517
for i in *.jpeg; 
do 
	new=$(printf "%d.txt" "$a"); 
	echo 0 > "$new"; 
	let a=a+1; 
done