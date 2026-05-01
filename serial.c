#include <stdio.h>
#include <fcntl.h>
#include <unistd.h>
#include <termios.h>
#include <stdint.h>
#include <stdlib.h>

const uint32_t SYNC = 0xBA0BABED;

#define N 256
#define SZ N * (sizeof(float) + sizeof(float) + sizeof(char))


int main() {
    int fd = open("/dev/ttyUSB0", O_RDWR | O_NOCTTY);

    struct termios tty;
    tcgetattr(fd, &tty);

    cfmakeraw(&tty);                 // put terminal in raw mode
    cfsetispeed(&tty, B1000000);     // input baud
    cfsetospeed(&tty, B1000000);     // output baud

    tty.c_cflag |= (CLOCAL | CREAD); // enable receiver, ignore modem lines

    tcsetattr(fd, TCSANOW, &tty);

    uint32_t chunk = 0;

    void *buffer = malloc(SZ);

    while (1) {
        /* find sync */
        while (1) {
            uint8_t b;
            read(fd, &b, 1);

            chunk = (chunk >> 8) | ((uint32_t)b << 24);

            if (chunk == SYNC)
                break;
        }

        /* read frame */
        // float data[N * 2];
        char iref[N];
        size_t got = 0;
        while (got < SZ) {
            got += read(fd, ((uint8_t*)buffer) + got, SZ - got);
        }

        /* forward */
        write(STDOUT_FILENO, buffer, SZ);


        // for (int i = 0; i < N; i++) {
        //     float x = data[2*i];
        //     float y = data[2*i+1];

        //     printf("%f %f\n", x, y);
        // }

        // fflush(stdout);
    }
}