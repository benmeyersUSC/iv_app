TM_SYMBOLS = {
        "S0": ' ',
        "S1": '0',
        "S2": '1',
        "S3": 'R',
        "S4": 'L',
        "S5": 'N',
        "S6": 'D',
        "S7": 'A',
        "S8": 'S',
        "S9": ';',
        "S10": ':',
        "SENTINEL": '@',
        "V1": 'x',
        "V2": 'y',
        "V3": 'z',
        "V4": 'r',
        "V5": 'u',
        "V6": 'v',
        "V7": 's',
        "V8": 't',
        "V9": 'w',
        "ASTR": "*"
    }
TM_MAX_TAPE = 999

TM_SIGS = set()
TM_USED = set()

def specPrint(x):
    if __name__ == "__main__":
        print(x)

class Configuration:
    def __init__(self, readSymbol, writeSymbol, moveDirection, nextState):
        self.readSymbol = readSymbol
        self.writeSymbol = writeSymbol
        self.moveDirection = moveDirection
        self.nextState = nextState

    def getReadSymbol(self):
        return self.readSymbol

    def getWriteSymbol(self):
        return self.writeSymbol

    def getMoveDirection(self):
        return self.moveDirection

    def getNextState(self):
        return self.nextState
    def getNextSignature(self):
        return self.nextState + "-" + self.readSymbol

    def __str__(self):
        return f"WRITE: {self.getWriteSymbol()}, MOVE: {self.getMoveDirection()}, NEXT: {self.getNextState()}"

    def __repr__(self):
        return self.__str__()


class Tape:
    def __init__(self, size=54, tapeFill="S0"):
        self.head = 0
        self.size = size
        self.tapeFill = tapeFill
        self.values = [None] * self.size
        for i in range(size):
            self.values[i] = TM_SYMBOLS[self.tapeFill]

    def read(self):
        val = self.values[self.head]
        for k, s in TM_SYMBOLS.items():
            if s == val:
                return k
        return "S0"

    def write(self, s):
        self.values[self.head] = TM_SYMBOLS[s]

    def right(self):
        if self.head + 1 == self.size:
            self.size += 10
            old = self.values.copy()
            self.values = [None] * self.size
            for i in range(self.size):
                if i < self.size - 10:
                    self.values[i] = old[i]
                else:
                    self.values[i] = TM_SYMBOLS[self.tapeFill]
        self.head += 1

    def left(self):
        if self.head == 0:
            self.size += 10
            old = self.values.copy()
            self.values = [None] * self.size
            for i in range(self.size):
                if i < 10:
                    self.values[i] = TM_SYMBOLS[self.tapeFill]
                else:
                    self.values[i] = old[i-10]
            self.head = 9
        else:
            self.head -= 1

    def __str__(self):
        ret = ""
        for i in range(self.size):
            if i % 81 == 0:
                ret += "\n"
                if i == 0:
                    ret += "|"
                else:
                    ret += "|"
            if i == self.head:
                # ret += f"<<<{self.values[i]}>>>|"
                # ret += f"\033[92m<<<{self.values[i]}>>>\033[0m|"
                if __name__ == "__main__":
                    ret += f"\033[30;42m<<<{self.values[i]}>>>\033[0m|"
                else:
                    # ret += f'<span class="highlight-green"><<<{self.values[i]}>>></span>|'
                    ret += f'<span class="highlight-green"><<<</span>' + self.values[i] + f'<span class="highlight-green">>>></span>|'
            else:
                ret += f"{self.values[i]}|"

        if len(ret) > 294:
            if __name__ == "__main__":
                ret = (ret[:148] +                "\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t***\n" +

               "\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t***\n" +
               "\t\t\t\t\t\t\t\t\t******************************************************************\n" +
                "\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t***\n" +
               ret[-161:].replace("\n", ""))
            else:
                ret = (ret[:148] +
                     "\n\t\t\t\t\t                        **************                            \n" +
                       "\t\t\t\t\t            ***************************************               \n" +
                       "\t\t\t\t\t******************************************************************\n" +
                       "\t\t\t\t\t            ***************************************               \n" +
                       "\t\t\t\t\t                        **************                            \n" +
                       ret[-180:].replace("\n", ""))

        return ret

    def __repr__(self):
        return self.__str__()

    def size(self):
        return self.size


class TuringMachine:

    def __init__(self, tape: Tape, sizeLimit=TM_MAX_TAPE, description=None, printConfigs=False):
        self.head = {}
        self.currentState = None
        self.tape = tape
        self.sizeLimit = sizeLimit
        self.printConfigs = printConfigs

        if description != None:
            if description[-11:] == ".javaturing":
                with open(description, "r") as fn:
                    description = fn.read().strip()
        if "#########" in description:
            description = description.split("#########")[0]

        foundInit = False

        for line in description.split(';'):
            line = line.strip()
            if len(line) == 0:
                continue
            parts = line.split('-')
            if len(parts) != 5:
                raise Exception(f"Invalid configuration format: {line}")

            try:
                state = parts[0].strip()
                readSymbol = parts[1].strip()
                writeSymbol = parts[2].strip()
                if parts[3].strip() == 'R':
                    direction = "RIGHT"
                elif parts[3].strip() == 'L':
                    direction = "LEFT"
                else:
                    direction = "STAY"
                nextState = parts[4].strip()

                if not foundInit:
                    self.currentState = state
                    foundInit = True

                configuration = Configuration(readSymbol, writeSymbol, direction, nextState)
                self.addConfiguration(state, configuration)

            except Exception as e:
                raise Exception(f"Invalid symbol in configuration: {line}\n{e}")

        if self.printConfigs:
            specPrint(f"Machine Program:")
            for k, v in self.head.items():
                specPrint(f"\t{k}:")
                for c, b in v.items():
                    TM_SIGS.add(f"{k}-{c}")
                    specPrint(f"\t\t{c}: {b}")

    def addConfiguration(self, state, configuration):
        if state not in self.head:
            self.head[state] = {}
        if configuration.getReadSymbol() in self.head[state]:
            raise Exception(
                f"Duplicate configuration for state {state} and read symbol {configuration.getReadSymbol()}")
        self.head[state][configuration.getReadSymbol()] = configuration

    def run(self, show=True, start=0, saveFirst=27):
        """
        Primary method to run machine until limit, has all checking measures
        :param show: boolean to specPrint tape at end or not
        :return: None
        """
        first_steps = []

        steps = start

        first_steps.append(f"\nStep {steps}:\tHEAD: {self.getTape().head}: ON STATE: {self.currentState}({self.tape.read()} = \"{TM_SYMBOLS[self.tape.read()]}\") {self.getTape()}")
        # first_steps.append(str(self.getTape()))

        while self.currentState != "HALT":
            if self.tape.size >= self.sizeLimit:
                if show:
                    specPrint(self.getTape())
                specPrint(f"Halting: tape size limit {self.sizeLimit} reached")
                specPrint(f"Steps taken: {steps}")
                return "\n\n".join(first_steps)

            steps = self.makeStep(steps)

            if steps < saveFirst:
                first_steps.append(f"\nStep {steps+1}:\tHEAD: {self.getTape().head}: ON STATE: {self.currentState}({self.tape.read()} = \"{TM_SYMBOLS[self.tape.read()]}\") {self.getTape()}")
            elif steps == saveFirst:
                first_steps.append("\n...........")
            steps += 1

        if show:
            specPrint(self.getTape())
        specPrint(f"Machine halted normally after {steps} steps")

        return "\n\n".join(first_steps)

    def runStepwise(self, _steps=0, every=1):
        """
        For debugging and visualization, step through program stepwise
        :return:
        """
        steps = _steps

        specPrint(f"\nStep {steps}:\tHEAD: {self.getTape().head}: ON STATE: {self.currentState}({self.tape.read()} = \"{TM_SYMBOLS[self.tape.read()]}\")")
        specPrint(self.getTape())

        while self.currentState != "HALT":
            if self.tape.size >= self.sizeLimit:
                specPrint(self.getTape())
                specPrint(f"Halting: tape size limit {self.sizeLimit} reached")
                specPrint(f"Steps taken: {steps}")
                return

            steps = self.makeStep(steps)
            if steps % every == 0:
                userIn = input()
                if "run" in userIn.lower():
                    runTo = input("to step (integer or \"end\") >> ")
                    if "end" in runTo.lower():
                        return self.run(show=True, start=steps)
                    else:
                        try:
                            self.runTo(stop=int(runTo.strip()))
                            return self.runStepwiseFrom(start=int(runTo.strip()), every=every)
                        except Exception as e:
                            specPrint(f"Need to enter integer or \"end\", running to end")
                            return self.run(start=steps)


                specPrint(f"\nStep {steps}:\tHEAD: {self.getTape().head}: ON STATE: {self.currentState}({self.tape.read()} = \"{TM_SYMBOLS[self.tape.read()]}\")")
                specPrint(self.getTape())

            steps += 1

        specPrint(self.getTape())
        specPrint(f"Machine halted normally after {steps} steps")

    def runTo(self, stop=None, show=True):
        if stop is None:
            return self.run()
        steps = 0

        while self.currentState != "HALT":
            if self.tape.size >= self.sizeLimit:
                if show:
                    specPrint(self.getTape())
                specPrint(f"Halting: tape size limit {self.sizeLimit} reached")
                specPrint(f"Steps taken: {steps}")
                return
            elif steps >= stop:
                if show:
                    specPrint(self.getTape())
                specPrint(f"Halting: step limit {stop} reached")
                specPrint(f"Steps taken: {steps}")
                return

            steps = self.makeStep(steps)

            steps += 1

        if show:
            specPrint(self.getTape())
        specPrint(f"Machine halted normally after {steps} steps")

    def runStepwiseFrom(self, start=0, every=1):
        if start == 0:
            return self.runStepwise()

        self.runTo(start)
        self.runStepwise(_steps=start+1, every=every)

    def getTape(self):
        return self.tape


    def printUnary(self, tape=True):
        # Initialize a string to accumulate output
        output = ""

        # Convert tape values to symbols for easier reading
        symbols = []
        for i, val in enumerate(self.tape.values):
            if i == self.tape.head:
                symbols.append(f"[{val}]")  # Mark current head position
            else:
                symbols.append(val)

        # Find unary numbers (patterns of 1s surrounded by 0s)
        tape_str = ''.join(symbols)
        numbers = []
        numSet = set()
        current_number = []
        in_number = False



        for i, symbol in enumerate(symbols):
            actual_symbol = symbol.strip('[]')  # Remove head position markers if present

            if actual_symbol == '1':
                current_number.append(symbol)
                in_number = True
            elif in_number and actual_symbol == '0':
                if current_number:  # Only add if we found some 1s
                    numbers.append(current_number)
                    numSet.add(len(current_number))
                    current_number = []
                in_number = False
            elif actual_symbol == '2':  # Include 2s in output for visibility
                if not in_number:
                    current_number = []
                current_number.append(symbol)
                in_number = True

        if len(numSet) <= 1:
            output = f"There are no unary digits on this tape couched by 0s" if len(numSet) == 0 else f"\"1\" is repeated in unary redundantly ({len(numbers)} times)..."
            specPrint(output)
            return output
        # Add each unary number to the output string
        output += "\nUnary Numbers on Tape (in form |0| |number| |0|):\n"
        output += "-" * 40 + "\n"
        for i, num in enumerate(numbers, 1):
            # Count actual 1s and 2s (excluding head markers)
            value = sum(1 for s in num if s.strip('[]') in ['1', '2'])
            output += f"{value}: {''.join(num) if len(num) < 24 else f"111... ({len(num)-6} x '1') ...111"}\n"
        output += "-" * 40 + "\n"
        output += f"Head Position: {self.tape.head}\n"

        if tape:
            # Also add the full tape to the output for reference
            output += "\nFull Tape:\n"
            output += str(self.tape) + "\n"

        # Return the full formatted output string
        specPrint(output)
        return output


    def showConfigurationsUsed(self):
        specPrint(len(TM_SIGS))
        specPrint(len(TM_USED))
        diff = list(TM_SIGS - TM_USED)
        diff.sort()
        for x in diff:
            specPrint(x.replace("-", " - "))
        return diff

    def makeStep(self, steps):
        currentSymbol = self.tape.read()
        try:
            configuration = self.head[self.currentState][currentSymbol]
        except Exception as e:
            specPrint(self.getTape())
            raise Exception(f"{e}....No signature found: {self.currentState}-{currentSymbol}\nSteps: {steps}")

        if configuration is None:
            specPrint(f"No configuration defined for state: {self.currentState} and symbol: {currentSymbol}")
            specPrint(f"Steps taken: {steps}")
            return steps

        self.tape.write(configuration.getWriteSymbol())

        if configuration.getMoveDirection() == "LEFT":
            self.tape.left()
        elif configuration.getMoveDirection() == "RIGHT":
            self.tape.right()

        TM_USED.add(f"{self.currentState}-{currentSymbol}")
        self.currentState = configuration.getNextState()

        return steps


if __name__ == "__main__":
    # isOdd = TuringMachine(Tape(), description="isOdd.javaturing")
    # isOdd.run()

    # art = TuringMachine(Tape(tapeFill="ASTR"), description="art.javaturing")
    # art.run()
    #
    # counting = TuringMachine(Tape(tapeFill="S0"), description="counting.javaturing", sizeLimit=10305, printConfigs=True)
    # counting.run(show=False)
    # counting.showConfigurationsUsed()
    # counting.printUnary(tape=False)

    # counting = TuringMachine(Tape(), description="counting.javaturing", sizeLimit=549, printConfigs=False)
    # print(counting.run(saveFirst=101))
    
    sqrt2 = TuringMachine(Tape(), description="sqrt2.javaturing", sizeLimit=200, printConfigs=True)
    # sqrt2.runStepwiseFrom(start=2900)
    sqrt2.run()
    # sqrt2.showConfigurationsUsed()


    # doubling = TuringMachine(Tape(), description="doubling.javaturing", sizeLimit=549, printConfigs=False)
    # specPrint(doubling.run(saveFirst=69))
    # doubling.runStepwise()

    # # run to a certain step, then stepwise from there
    # doubling.runStepwiseFrom(start=13)

    # # equal to this, but first is preferred
    # doubling.runTo(stop=400)
    # doubling.runStepwise(_steps=400)


    # # Show configurations used or not
    # doubling.showConfigurationsUsed()
    # machine.printUnary(tape=False)



