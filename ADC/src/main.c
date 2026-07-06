#include "stm32f1xx.h"
#include "FreeRTOS.h"
#include "task.h"

extern uint32_t SystemCoreClock; 

void RCC_Config(void);
void GPIO_Config(void);
void USART1_Config(void);
void ADC1_Config(void);
uint16_t ADC1_Read(void);
void USART1_SendChar(char c);

void vTaskADCReadAndSend(void *pvParameters);

int main(void) {
    SystemCoreClock = 8000000;  // Informar ao sistema que o clock principal estará rodando a 8 MHz.
    RCC_Config();
    GPIO_Config();
    USART1_Config();
    ADC1_Config();

    xTaskCreate(vTaskADCReadAndSend, "ADC_Task", 128, NULL, 1, NULL);   // Criação da tarefa (rotina independente) baseada na função vTaskADCReadAndSend
    vTaskStartScheduler();

    while (1) {}
}

// --- Tarefa do FreeRTOS (Leitura do sensor + transimissão do dados) ---
void vTaskADCReadAndSend(void *pvParameters) {
    while (1) {
        uint16_t adc = ADC1_Read();

        // Quebra os 12 bits em 2 pacotes indestrutíveis:
        // Byte 1: Força o bit mais significativo a ser '1' (0x80) e guarda os 5 bits do topo
        uint8_t byte1 = 0x80 | ((adc >> 7) & 0x1F);
        
        // Byte 2: Força o bit mais significativo a ser '0' e guarda os 7 bits de baixo
        uint8_t byte2 = adc & 0x7F;

        USART1_SendChar((char)byte1);
        USART1_SendChar((char)byte2);
    }
}

// --- Funções de Configuração ---
void RCC_Config(void) {
    RCC->APB2ENR |= RCC_APB2ENR_IOPAEN | RCC_APB2ENR_IOPBEN | RCC_APB2ENR_AFIOEN;      // IOPAEN : 1 (O port A clock enabled) | IOPBEN : 1 (O port B clock enabled)  | AFIOEN : 1 (Alternate function IO clock enable)
    RCC->APB2ENR |= RCC_APB2ENR_USART1EN | RCC_APB2ENR_ADC1EN;                         // USART1EN : 1 (USART1 clock enable)  | ADC1EN : 1 (ADC 1 interface clock enable)
    RCC->CFGR &= ~RCC_CFGR_ADCPRE;
    RCC->CFGR |= RCC_CFGR_ADCPRE_DIV6;                                                 // ADCPRE : 10 (ADC prescaler) ---> PCLK2 divided by 6  ---. Clock de 8M/6  = 1.333 MHz
}

void GPIO_Config(void) {
    GPIOA->CRH &= ~(GPIO_CRH_CNF9 | GPIO_CRH_MODE9);
    GPIOA->CRH |= (GPIO_CRH_CNF9_1 | GPIO_CRH_MODE9_0);     // PA9 (TX)  ---> MODE:01 (Output mode, max speed 10 MHz) | CNF:10 (Alternate function output Push-pull) 
    GPIOA->CRH &= ~(GPIO_CRH_CNF10 | GPIO_CRH_MODE10);      
    GPIOA->CRH |= GPIO_CRH_CNF10_0;                         // PA10 (RX) ---> MODE:00 (Input mode (reset state)) | CNF:01 (Floating input (reset state))
    GPIOB->CRL &= ~(GPIO_CRL_CNF1 | GPIO_CRL_MODE1);        // PB1 (ADC read) ---> MODE:00 (Input mode (reset state)) | CNF:00 (Analog mode)  
}

void USART1_Config(void) {
    USART1->BRR = 0x45;                                     // 115200 bps
    USART1->CR1 |= (USART_CR1_TE | USART_CR1_RE | USART_CR1_UE);
}

void ADC1_Config(void) {
    ADC1->CR2 |= ADC_CR2_ADON;                  // Wakes up the ADC from Power Down mode
    for (volatile int i = 0; i < 10000; i++);
    ADC1->SMPR2 |= (7 << 27);                   // SMPx[2:0]: Channel x Sample time selection --> SMP9[2:0] --> 111: 239.5 cycles 
                                                // The total conversion time is Tconv = (239.5 + 12.5) * (1/(1.333 MHz)) = 189.05 us 
                                                // The Data conversin rate = 5289.6 SPS
    ADC1->CR2 |= ADC_CR2_CAL;                   // CAL : A/D Calibration ---> (0: Calibration completed  |  1: Enable calibration)
    while (ADC1->CR2 & ADC_CR2_CAL);            // Cheking Calibration 
    ADC1->SQR1 &= ~ADC_SQR1_L;                  // L[3:0]: Regular channel sequence length ---> 0000: 1 conversion
    ADC1->SQR3 = 9;                             // SQ1[4:0]: first conversion in regular sequence ---> channel 9 (01001) 
}

uint16_t ADC1_Read(void) {
    ADC1->CR2 |= ADC_CR2_ADON;                  // Start conversion 
    while (!(ADC1->SR & ADC_SR_EOC));           // Checking if the EOC (End of Conversion) indicator has been set
    return (uint16_t)(ADC1->DR & 0xFFF);        // Storing data conversion
}

void USART1_SendChar(char c) {
    while (!(USART1->SR & USART_SR_TXE));
    USART1->DR = c;
}

