figure;
surf(av)

figure;
subplot(2,4,1)
plot(av(6,:))
subplot(2,4,2)
plot(av(13,:))
subplot(2,4,3)
plot(av(8,:))
subplot(2,4,4)
plot(av(11,:))
subplot(2,4,5)
plot(av(22,:))
subplot(2,4,6)
plot(av(23,:))
subplot(2,4,7)
plot(av(29,:))
subplot(2,4,8)
plot(av(30,:))

figure;
plot(mean(av))
